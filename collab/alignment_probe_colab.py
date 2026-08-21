"""
Cross-lingual alignment probing on Google Colab.

Same measurement as scripts/diagnostics/crosslingual_alignment.py and
layerwise_alignment.py, but pulling checkpoints from the HF backup repos
instead of scratch, so it runs without the HPC.

Each "# %% CELL n" marker below is one Colab cell - paste them in order.

One sweep covers both views: every checkpoint x every encoder layer,
with the final layer also written out for the trajectory plots.

Disk discipline matters here: 20 checkpoints x ~2.5GB would fill Colab's
disk, so each checkpoint is downloaded, used, and deleted before the next.
"""

# %% CELL 1 - install (pins match the HPC lockfile)
#
# datasets<4 is the important one: Colab ships 4.x, which REMOVED script-based
# loaders, and Muennighoff/flores200 is script-based. The HPC pyproject pins
# datasets>=3.0.0,<4.0.0 (resolves to 3.6.0), which is why it works there.
#
# !pip install -q "datasets>=3.0.0,<4.0.0" "transformers>=5.12.1" "huggingface_hub>=0.24" matplotlib
#
# THEN: Runtime > Restart session, and continue from CELL 2.
# (datasets is already imported at Colab startup, so the new version only
# takes effect after a restart. You do not need to re-run CELL 1 after it.)

# %% CELL 2 - login. Not needed: every repo above is public.
# from huggingface_hub import notebook_login
# notebook_login()

# %% CELL 3 - imports and configuration
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# base model on the Hub (step 0), and the backup repo holding checkpoints
MODELS = {
    # phase 1: monolingual isiXhosa CPT
    "t5":         ("google-t5/t5-large", "ChonkeyJellyfish/cpt-xhosa-t5-large"),
    "byt5":       ("google/byt5-large",  "ChonkeyJellyfish/cpt-xhosa-byt5-large"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "ChonkeyJellyfish/cpt-xhosa-nguni-byt5-large"),
    # phase 2: bilingual isiXhosa + English CPT. Same base model, same schedule,
    # same step budget - the only difference is English mixed into the corpus,
    # so step 0 is identical to the phase-1 arm and the curves start together.
    "byt5-bilingual":       ("google/byt5-large",
                             "ChonkeyJellyfish/cpt-bilingual-byt5-large"),
    "t5-bilingual":         ("google-t5/t5-large",
                             "ChonkeyJellyfish/cpt-bilingual-t5-large"),
    "nguni-byt5-bilingual": ("francois-meyer/nguni-byt5-large",
                             "ChonkeyJellyfish/cpt-bilingual-nguni-byt5-large"),
}
ALL_STEPS = [0] + list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))

RUN_MODELS = ["byt5-bilingual"]    # then ["t5-bilingual"] in a second pass
BATCH_SIZE = 16                    # drop to 8 if the byte models OOM
DTYPE = torch.float32              # matches the HPC runs; do not change
OUT = Path("/content/alignment")
CKPT_DIR = Path("/content/_ckpt")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# %% CELL 4 - load FLORES-200 devtest (1,012 aligned pairs)
#
# Same call as src/data_processing/download_finetune_data.py on the HPC, so
# the sentences are identical. Needs datasets<4 from CELL 1 (and a restart).
from datasets import load_dataset

FLORES_DATASET = "Muennighoff/flores200"


def load_flores_pairs(src="eng_Latn", tgt="xho_Latn", split="devtest"):
    def sentences(lang):
        ds = load_dataset(FLORES_DATASET, lang, split=split, trust_remote_code=True)
        col = next((c for c in ds.column_names if c.startswith("sentence")), None)
        if col is None:
            raise ValueError(f"no sentence column in {ds.column_names}")
        return [s.strip() for s in ds[col]]
    return sentences(src), sentences(tgt)


def load_flores_pairs_tarball(src="eng_Latn", tgt="xho_Latn", split="devtest"):
    """Fallback if you would rather not pin datasets: the official NLLB
    release, same underlying files the HF wrapper serves."""
    import tarfile
    import urllib.request

    root = Path("/content/flores")
    if not root.exists():
        tgz = Path("/content/flores200.tar.gz")
        if not tgz.exists():
            print("downloading FLORES-200 (~25MB)...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", tgz)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(root)

    def sentences(lang):
        matches = sorted(root.rglob(f"{lang}.{split}"))
        if not matches:
            raise FileNotFoundError(f"{lang}.{split} not found under {root}")
        return [line.strip() for line in
                matches[0].read_text(encoding="utf-8").split("\n") if line.strip()]
    return sentences(src), sentences(tgt)


ENG, XHO = load_flores_pairs()
assert len(ENG) == len(XHO), "FLORES sides must be the same length"
print(f"{len(ENG)} pairs")
print(f"  eng: {ENG[0][:90]}")
print(f"  xho: {XHO[0][:90]}")

# %% CELL 5 - the measurement (identical to the HPC implementation)


@torch.no_grad()
def encode(encoder, tokenizer, texts, batch_size, device):
    """Mean-pooled final hidden states over non-pad tokens, L2-normalised."""
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)
        hidden = encoder(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_layers(encoder, tokenizer, texts, batch_size, device):
    """Same pooling, but at every hidden layer (layer 0 = embedding output)."""
    per_layer = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)
        out = encoder(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for i, hidden in enumerate(out.hidden_states):
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1).cpu()
            if i >= len(per_layer):
                per_layer.append([])
            per_layer[i].append(pooled)
    return [torch.cat(chunks) for chunks in per_layer]


def alignment_stats(eng, xho):
    """Idris et al.: cosine_mean = mean diagonal of M; cosine_gap (Eq. 1) =
    cosine_mean - mean over ALL N^2 entries; P@1 both directions."""
    sims = eng @ xho.T
    n = sims.shape[0]
    cosine_mean = sims.diagonal().mean().item()
    baseline = sims.mean().item()
    gap = cosine_mean - baseline
    r_e2x = (sims.argmax(dim=1) == torch.arange(n)).float().mean().item()
    r_x2e = (sims.argmax(dim=0) == torch.arange(n)).float().mean().item()
    return {"cosine_mean": cosine_mean, "baseline": baseline, "cosine_gap": gap,
            "p_at_1_e2x": r_e2x, "p_at_1_x2e": r_x2e}


# %% CELL 6 - fetch one checkpoint, use it, delete it


def fetch_checkpoint(model_name, step):
    """Return a local path for this step. Step 0 = the Hub base model."""
    base_id, repo_id = MODELS[model_name]
    if step == 0:
        return base_id
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    snapshot_download(repo_id=repo_id, allow_patterns=[f"checkpoint-{step}/*"],
                      local_dir=str(CKPT_DIR))
    return str(CKPT_DIR / f"checkpoint-{step}")


def release():
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def done_steps(path):
    """Resume support: Colab disconnects, so skip what is already written."""
    if not path.exists():
        return set()
    return {json.loads(l)["step"]
            for l in path.read_text().split("\n") if l.strip()}


# %% CELL 7 - THE SWEEP: every checkpoint, every encoder layer
#
# One pass gives both views. hidden_states[-1] IS last_hidden_state for T5
# (both are taken after the final layer norm), so the last layer's row is
# also written to {model}.jsonl for the trajectory plots - no second pass,
# no second download. That final layer norm is also why the top layer
# looks geometrically unlike the ones below it.
for model_name in RUN_MODELS:
    base_id, _ = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    layers_path = OUT / f"{model_name}_layers.jsonl"
    traj_path = OUT / f"{model_name}.jsonl"
    already = done_steps(layers_path)

    print(f"\n=== {model_name}  ({len(already)}/{len(ALL_STEPS)} steps already done)")
    for step in ALL_STEPS:
        if step in already:
            continue
        t0 = time.time()
        path = fetch_checkpoint(model_name, step)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=DTYPE).to(DEVICE)
        encoder = model.get_encoder().eval()

        eng_layers = encode_layers(encoder, tokenizer, ENG, BATCH_SIZE, DEVICE)
        xho_layers = encode_layers(encoder, tokenizer, XHO, BATCH_SIZE, DEVICE)
        n_layers = len(eng_layers)
        print(f"\nstep {step}  ({n_layers} layers, {time.time() - t0:.0f}s to encode)")
        print(f"{'layer':>6} {'cos_mean':>9} {'baseline':>9} {'cos_gap':>8} "
              f"{'P@1 e>x':>8} {'P@1 x>e':>8}")

        with open(layers_path, "a") as f:
            for layer, (e, x) in enumerate(zip(eng_layers, xho_layers)):
                stats = alignment_stats(e, x)
                f.write(json.dumps({"model": model_name, "step": step, "layer": layer,
                                    **stats, "n_pairs": len(ENG)}) + "\n")
                # 20 checkpoints x 37 layers would be 740 printed lines, so
                # only show first / middle / last; the jsonl has everything.
                if layer in (0, n_layers // 2, n_layers - 1):
                    print(f"{layer:>6} {stats['cosine_mean']:>9.4f} "
                          f"{stats['baseline']:>9.4f} {stats['cosine_gap']:>8.4f} "
                          f"{stats['p_at_1_e2x']:>8.2%} {stats['p_at_1_x2e']:>8.2%}")
                if layer == n_layers - 1:
                    with open(traj_path, "a") as g:
                        g.write(json.dumps({"model": model_name, "step": step,
                                            **stats, "n_pairs": len(ENG)}) + "\n")

        del model, encoder, eng_layers, xho_layers
        release()

# %% CELL 9 - plots
import matplotlib.pyplot as plt
from collections import defaultdict

MODEL_ORDER = ["t5", "byt5", "nguni-byt5",
               "t5-bilingual", "byt5-bilingual", "nguni-byt5-bilingual"]
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


# trajectory: metric vs CPT step, one line per model, base as dashed reference
rows = [r for p in sorted(OUT.glob("*.jsonl")) if "_layers" not in p.name
        for r in read_jsonl(p)]
for key, label in [("cosine_gap", "Cosine gap"), ("p_at_1_e2x", "P@1 eng to xho"),
                   ("p_at_1_x2e", "P@1 xho to eng"), ("cosine_mean", "Cosine mean"),
                   ("baseline", "Baseline similarity")]:
    series = defaultdict(list)
    for r in rows:
        series[r["model"]].append((r["step"], r[key]))
    if not series:
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(series, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
        pts = sorted(series[m])
        c = f"C{MODEL_ORDER.index(m)}" if m in MODEL_ORDER else "C7"
        base = [v for s, v in pts if s == 0]
        cpt = [(s, v) for s, v in pts if s > 0]
        if base:
            ax.axhline(base[0], color=c, ls="--", lw=1.0, alpha=0.55)
        if cpt:
            ax.plot([s for s, _ in cpt], [v for _, v in cpt],
                    marker="o", ms=4, color=c, label=m)
    ax.set_xlabel("CPT step"); ax.set_ylabel(label); ax.set_title(label)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / f"{key}.png", dpi=150); plt.show()

# layer-wise: metric vs layer, one line per checkpoint
for p in sorted(OUT.glob("*_layers.jsonl")):
    model = p.stem.replace("_layers", "")
    lrows = read_jsonl(p)
    steps = sorted({r["step"] for r in lrows})
    colours = plt.cm.viridis([i / max(1, len(steps) - 1) for i in range(len(steps))])
    for key, label in [("cosine_gap", "Cosine gap"), ("p_at_1_e2x", "P@1 eng to xho"),
                       ("cosine_mean", "Cosine mean"), ("baseline", "Baseline similarity")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for step, colour in zip(steps, colours):
            pts = sorted((r["layer"], r[key]) for r in lrows if r["step"] == step)
            ax.plot([l for l, _ in pts], [v for _, v in pts], marker="o", ms=3,
                    color="black" if step == 0 else colour,
                    ls="--" if step == 0 else "-",
                    label="base" if step == 0 else f"step {step}")
        ax.set_xlabel("Encoder layer"); ax.set_ylabel(label)
        ax.set_title(f"{label} by layer for {model}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(PLOTS / f"{model}_{key}.png", dpi=150); plt.show()

# %% CELL 10 - save results off the VM before it disconnects
# from google.colab import files
# shutil.make_archive("/content/alignment_results", "zip", OUT)
# files.download("/content/alignment_results.zip")

# or mount Drive and copy:
# from google.colab import drive
# drive.mount("/content/drive")
# shutil.copytree(OUT, "/content/drive/MyDrive/alignment", dirs_exist_ok=True)

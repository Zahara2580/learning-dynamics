"""
Representation anisotropy on Google Colab: are token representations
collapsing onto a narrow cone, and does it differ by language?

DIFFERENT MEASUREMENT from alignment_probe_colab.py. That one is
sentence-level and cross-lingual: mean-pool each sentence, compare English
against its isiXhosa translation. This one is token-level and WITHIN
language: take token vectors from one language and ask how directionally
distinct they are from each other. English and isiXhosa are measured
separately and never compared to each other - the comparison is
English-against-English versus isiXhosa-against-isiXhosa, at the same layer
of the same checkpoint.

Definition, following the cosine-similarity metric in the anisotropy
literature: for layer l,

    CosSim(l) = mean over token pairs (i, j), i != j, of  h_i . h_j / (|h_i||h_j|)

Low values mean well-separated embeddings; high values mean the
representations have collapsed onto a narrow high-dimensional cone.

ONE DEVIATION, and it is in your favour. The published formulation samples
N = 1000 random pairs per layer and averages. We compute the mean over ALL
pairs instead, which is the quantity that sample estimates. For L2-normalised
vectors,

    sum_{i != j} v_i . v_j  =  ||sum_i v_i||^2 - n

so the exact mean is (||s||^2 - n) / (n(n-1)) from a running sum s and a
count n - O(d) memory, no token matrix held, and no sampling. With ~120,000
byte tokens per language the exact value costs less than storing the sample
would. Write it up as "computed over all pairs rather than a 1,000-pair
sample"; it is the same estimand.

Each "# %% CELL n" marker is one Colab cell - paste them in order.
Disk discipline as in alignment_probe_colab.py: download, use, delete.
"""

# %% CELL 1 - install (same pins as alignment_probe_colab.py)
#
# !pip install -q "datasets>=3.0.0,<4.0.0" "transformers>=5.12.1" "huggingface_hub>=0.24" matplotlib
#
# THEN: Runtime > Restart session, and continue from CELL 2.

# %% CELL 2 - imports and configuration
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    "t5":         ("google-t5/t5-large", "ChonkeyJellyfish/cpt-xhosa-t5-large"),
    "byt5":       ("google/byt5-large",  "ChonkeyJellyfish/cpt-xhosa-byt5-large"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "ChonkeyJellyfish/cpt-xhosa-nguni-byt5-large"),
    "byt5-bilingual":       ("google/byt5-large",
                             "ChonkeyJellyfish/cpt-bilingual-byt5-large"),
    "t5-bilingual":         ("google-t5/t5-large",
                             "ChonkeyJellyfish/cpt-bilingual-t5-large"),
    "nguni-byt5-bilingual": ("francois-meyer/nguni-byt5-large",
                             "ChonkeyJellyfish/cpt-bilingual-nguni-byt5-large"),
}

ALL_STEPS = [0] + list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
# the seven the figures plot - a third of the runtime, same story
FIGURE_STEPS = [0, 1000, 3000, 4000, 5000, 6000, 10000]

RUN_MODELS = ["t5"]                # then ["byt5"], then ["nguni-byt5"]
STEPS = ALL_STEPS                  # switch to FIGURE_STEPS if time is short
BATCH_SIZE = 16                    # drop to 8 if the byte models OOM
MAX_LENGTH = 512
DTYPE = torch.float32              # matches the HPC runs; do not change

OUT = Path("/content/anisotropy")
CKPT_DIR = Path("/content/_ckpt")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# %% CELL 3 - FLORES-200 devtest, the same 1,012 sentences the other probes use
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


ENG, XHO = load_flores_pairs()
assert len(ENG) == len(XHO), "FLORES sides must be the same length"
print(f"{len(ENG)} sentences per language")

# %% CELL 4 - the measurement


@torch.no_grad()
def anisotropy_layers(encoder, tokenizer, texts, batch_size, device,
                      max_length=MAX_LENGTH):
    """Mean pairwise cosine between token vectors, at every hidden layer.

    Returns (per_layer_cosine, n_tokens, n_truncated).

    Padding is excluded via the attention mask, and EOS is excluded on top of
    that. EOS is one near-identical vector appended to all 1,012 sentences;
    left in, it contributes ~1,012 mutually parallel vectors to every pool and
    inflates the measure by an amount that has nothing to do with the language.
    """
    drop = {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id)
            if i is not None}
    sums, n_tokens, n_trunc = None, 0, 0

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        n_trunc += int((enc["attention_mask"].sum(1) >= max_length).sum())

        keep = enc["attention_mask"].bool()
        for tid in drop:
            keep &= enc["input_ids"] != tid
        mask = keep.unsqueeze(-1).float()

        out = encoder(**enc, output_hidden_states=True)
        if sums is None:
            sums = [torch.zeros(out.hidden_states[0].shape[-1], dtype=torch.float64)
                    for _ in out.hidden_states]
        for i, hidden in enumerate(out.hidden_states):
            v = F.normalize(hidden.float(), dim=-1) * mask
            # per-batch sum in fp32 on device, accumulated in fp64 on CPU:
            # ~120k unit vectors summed straight into fp32 loses digits
            sums[i] += v.sum(dim=(0, 1)).double().cpu()
        n_tokens += int(keep.sum())

    cos = [((s.dot(s).item() - n_tokens) / (n_tokens * (n_tokens - 1)))
           for s in sums]
    return cos, n_tokens, n_trunc


# %% CELL 5 - fetch one checkpoint, use it, delete it


def fetch_checkpoint(model_name, step):
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
    """Colab disconnects; skip what is already written."""
    if not path.exists():
        return set()
    return {json.loads(l)["step"]
            for l in path.read_text().split("\n") if l.strip()}


# %% CELL 6 - TIME IT FIRST. One checkpoint, then extrapolate before committing.
#
# Run this before CELL 7. It downloads step 10000 of RUN_MODELS[0], measures
# it, and prints what the full sweep would cost at that rate.
_m = RUN_MODELS[0]
_tok = AutoTokenizer.from_pretrained(MODELS[_m][0])
_t0 = time.time()
_path = fetch_checkpoint(_m, 10000)
_t_dl = time.time() - _t0
_model = AutoModelForSeq2SeqLM.from_pretrained(_path, dtype=DTYPE).to(DEVICE)
_enc = _model.get_encoder().eval()
_t1 = time.time()
_cos, _n, _tr = anisotropy_layers(_enc, _tok, ENG, BATCH_SIZE, DEVICE)
_t_eng = time.time() - _t1
print(f"download {_t_dl:.0f}s   english encode {_t_eng:.0f}s   "
      f"{_n:,} tokens, {len(_cos)} layers, {_tr} truncated")
print(f"per checkpoint (both languages): ~{_t_dl + 2 * _t_eng:.0f}s")
print(f"{len(STEPS)} steps x {len(RUN_MODELS)} model(s): "
      f"~{(_t_dl + 2 * _t_eng) * len(STEPS) * len(RUN_MODELS) / 60:.0f} min")
print(f"all 3 models x {len(ALL_STEPS)} steps: "
      f"~{(_t_dl + 2 * _t_eng) * len(ALL_STEPS) * 3 / 3600:.1f} h  "
      f"(byte models are slower than t5 - treat this as a floor)")
del _model, _enc
release()

# %% CELL 7 - THE SWEEP
for model_name in RUN_MODELS:
    base_id, _ = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    out_path = OUT / f"{model_name}_anisotropy.jsonl"
    already = done_steps(out_path)

    print(f"\n=== {model_name}  ({len(already)}/{len(STEPS)} steps already done)")
    for step in STEPS:
        if step in already:
            continue
        t0 = time.time()
        path = fetch_checkpoint(model_name, step)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=DTYPE).to(DEVICE)
        encoder = model.get_encoder().eval()

        per_lang = {}
        for lang, texts in (("eng", ENG), ("xho", XHO)):
            per_lang[lang] = anisotropy_layers(encoder, tokenizer, texts,
                                               BATCH_SIZE, DEVICE)

        n_layers = len(per_lang["eng"][0])
        print(f"\nstep {step}  ({n_layers} layers, {time.time() - t0:.0f}s)  "
              f"tokens eng={per_lang['eng'][1]:,} xho={per_lang['xho'][1]:,}")
        print(f"{'layer':>6} {'eng':>9} {'xho':>9} {'xho-eng':>9}")

        with open(out_path, "a") as f:
            for layer in range(n_layers):
                row = {"model": model_name, "step": step, "layer": layer}
                for lang in ("eng", "xho"):
                    cos, n, trunc = per_lang[lang]
                    row[f"cos_sim_{lang}"] = cos[layer]
                    row[f"n_tokens_{lang}"] = n
                    row[f"n_truncated_{lang}"] = trunc
                row["cos_sim_delta"] = row["cos_sim_xho"] - row["cos_sim_eng"]
                f.write(json.dumps(row) + "\n")
                if layer in (0, n_layers // 2, n_layers - 1):
                    print(f"{layer:>6} {row['cos_sim_eng']:>9.4f} "
                          f"{row['cos_sim_xho']:>9.4f} {row['cos_sim_delta']:>+9.4f}")

        del model, encoder, per_lang
        release()

# %% CELL 8 - quick look. Pull the jsonl for the real figures.
import matplotlib.pyplot as plt
from collections import defaultdict

for model_name in RUN_MODELS:
    rows = [json.loads(l) for l in
            (OUT / f"{model_name}_anisotropy.jsonl").read_text().split("\n") if l.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, lang in zip(axes, ("eng", "xho")):
        per_step = defaultdict(list)
        for r in rows:
            if r["step"] in FIGURE_STEPS:
                per_step[r["step"]].append((r["layer"], r[f"cos_sim_{lang}"]))
        for step in sorted(per_step):
            pts = sorted(per_step[step])
            ax.plot([x for x, _ in pts], [y for _, y in pts],
                    label="base" if step == 0 else f"{step:,}",
                    ls="--" if step == 0 else "-", lw=1.6)
        ax.set_xlabel("Encoder layer")
        ax.set_title({"eng": "English", "xho": "isiXhosa"}[lang])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean pairwise cosine")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(model_name)
    fig.tight_layout()
    plt.show()

# %% CELL 9 - download the results
# from google.colab import files
# for m in RUN_MODELS:
#     files.download(str(OUT / f"{m}_anisotropy.jsonl"))

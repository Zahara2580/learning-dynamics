"""Measure crosslingual encoder alignment across CPT checkpoints and layers."""

# %pip install -q "datasets>=3.0.0,<4.0.0" "transformers>=4.44" "huggingface_hub>=0.24" torch matplotlib

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download, login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    't5': ('google-t5/t5-large', 'ChonkeyJellyfish/cpt-xhosa-t5-large'),
    'byt5': ('google/byt5-large', 'ChonkeyJellyfish/cpt-xhosa-byt5-large'),
    'nguni-byt5': ('francois-meyer/nguni-byt5-large', 'ChonkeyJellyfish/cpt-xhosa-nguni-byt5-large')
}
ALL_STEPS = [0] + list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
BATCH_SIZE = 16
DTYPE = torch.float32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLORES_DATASET = 'Muennighoff/flores200'


def load_flores_pairs(src='eng_Latn', tgt='xho_Latn', split='devtest'):
    """Load aligned English and isiXhosa FLORES sentences."""

    def sentences(lang):
        """Read one language in the requested FLORES split."""
        ds = load_dataset(FLORES_DATASET, lang, split=split, trust_remote_code=True)
        col = next((c for c in ds.column_names if c.startswith('sentence')), None)
        if col is None:
            raise ValueError(f'no sentence column in {ds.column_names}')
        return [s.strip() for s in ds[col]]
    return (sentences(src), sentences(tgt))


@torch.no_grad()
def encode_layers(encoder, tokenizer, texts, batch_size, device):
    """Mean-pool nonpadding states at each encoder layer and normalise vectors."""
    per_layer = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start:start + batch_size],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        out = encoder(**enc, output_hidden_states=True)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        for i, hidden in enumerate(out.hidden_states):
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1).cpu()
            if i >= len(per_layer):
                per_layer.append([])
            per_layer[i].append(pooled)
    return [torch.cat(chunks) for chunks in per_layer]


def alignment_stats(eng, xho):
    """Compute cosine similarity and cosine gap """
    sims = eng @ xho.T
    n = sims.shape[0]
    cosine_mean = sims.diagonal().mean().item()
    baseline = sims.mean().item()
    gap = cosine_mean - baseline

    return {
        'cosine_mean': cosine_mean,
        'cosine_gap': gap,
        
    }


def run_sweep(output_dir: Path, models: list[str], steps: list[int]) -> None:
    """Save layer metrics and final-layer trajectories for aligned FLORES pairs."""
    eng, xho = load_flores_pairs()
    assert len(eng) == len(xho), "FLORES sides must have the same length"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        base, repo = MODELS[name]
        tokenizer = AutoTokenizer.from_pretrained(base)
        layers_path = output_dir / f"{name}_layers.jsonl"
        trajectory_path = output_dir / f"{name}.jsonl"
        completed = set()
        if trajectory_path.exists():
            completed = {json.loads(line)["step"] for line in trajectory_path.read_text().splitlines() if line.strip()}
        for step in steps:
            if step in completed:
                continue
            started = time.time()
            with tempfile.TemporaryDirectory(prefix="alignment_cpt_") as cache:
                path = base
                if step:
                    snapshot_download(repo_id=repo, allow_patterns=[f"checkpoint-{step}/*"], local_dir=cache)
                    path = str(Path(cache) / f"checkpoint-{step}")
                model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=DTYPE).to(DEVICE)
                encoder = model.get_encoder().eval()
                eng_layers = encode_layers(encoder, tokenizer, eng, BATCH_SIZE, DEVICE)
                xho_layers = encode_layers(encoder, tokenizer, xho, BATCH_SIZE, DEVICE)
                rows = [
                    dict(model=name, step=step, layer=layer, **alignment_stats(e, x), n_pairs=len(eng)) for layer,
                    (e, x) in enumerate(zip(eng_layers, xho_layers))
                ]
                existing = []
                if layers_path.exists():
                    existing = [json.loads(line) for line in layers_path.read_text().splitlines() if line.strip()]
                existing = [row for row in existing if row["step"] != step]
                temporary = layers_path.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(row) + "\n" for row in existing + rows))
                temporary.replace(layers_path)
                final = {key: value for key, value in rows[-1].items() if key != "layer"}
                with trajectory_path.open("a") as stream:
                    stream.write(json.dumps(final) + "\n")
                del model, encoder, eng_layers, xho_layers
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            print(f"{name} step {step}: {len(rows)} layers, {time.time() - started:.0f}s")


def main() -> None:
    """Run the requested alignment checkpoints."""
    parser = argparse.ArgumentParser(description="Crosslingual alignment on FLORES devtest")
    parser.add_argument("--output-dir", type=Path, default=Path("results/crosslingual_alignment"))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--steps", nargs="+", type=int, choices=ALL_STEPS, default=ALL_STEPS)
    parser.add_argument("--login", action="store_true", help="Prompt for Hugging Face access.")
    args = parser.parse_args()
    if args.login:
        from getpass import getpass
        login(token=getpass("Hugging Face token: "), add_to_git_credential=False)
    run_sweep(args.output_dir, args.models, args.steps)


if __name__ == "__main__":
    main()

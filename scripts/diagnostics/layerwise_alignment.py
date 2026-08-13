"""
Layer-wise cross-lingual alignment: the alignment probe swept across
every encoder layer, for a handful of evenly spaced checkpoints.

Same measurement as crosslingual_alignment.py - mean-pool over non-pad
positions, L2-normalise, paired/random cosine, margin, retrieval@1 - but
computed at EVERY encoder hidden layer (layer 0 = embedding output,
last = final encoder layer), answering "where in the network does
alignment live and how does CPT move it".

One row per (model, step, layer) -> results/alignment_layers/{model}.jsonl.

Usage:
    uv run python3 -m scripts.diagnostics.layerwise_alignment --model byt5
    uv run python3 -m scripts.diagnostics.layerwise_alignment --model t5 \
        --steps 0 5000 10000 --limit 200
"""

import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.diagnostics.crosslingual_alignment import (
    FLORES_DIR,
    MODELS,
    SPLIT,
    alignment_stats,
)

DEFAULT_STEPS = [0, 2000, 5000, 8000, 10000]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Per-layer encoder alignment.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS))
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out-dir", type=str, default="results/alignment_layers")
    parser.add_argument("--base-override", type=str, default=None)
    parser.add_argument("--checkpoints-override", type=str, default=None)
    parser.add_argument("--data-override", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def encode_layers(encoder, tokenizer, texts: list[str], batch_size: int, device) -> list[torch.Tensor]:
    """Mean-pooled, L2-normalised sentence vectors at every hidden layer.
    Returns one (n, d) tensor per layer, layer 0 = embedding output."""
    per_layer: list[list[torch.Tensor]] = []
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


def load_pairs(data_path: str, limit: int | None) -> tuple[list[str], list[str]]:
    if str(data_path).endswith(".jsonl"):
        rows = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
        eng = [r["source"].strip() for r in rows]
        xho = [r["target"].strip() for r in rows]
    else:
        from datasets import load_from_disk
        ds = load_from_disk(data_path)[SPLIT]
        eng = [s.strip() for s in ds["source"]]
        xho = [t.strip() for t in ds["target"]]
    if limit:
        eng, xho = eng[:limit], xho[:limit]
    return eng, xho


def main() -> None:
    args = parse_args()
    base_id, ckpt_root = MODELS[args.model]
    if args.base_override:
        base_id = args.base_override
    root = Path(args.checkpoints_override or ckpt_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    eng_texts, xho_texts = load_pairs(args.data_override or FLORES_DIR, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(base_id)

    ckpts = {int(m.group(1)): str(p) for p in root.glob("checkpoint-*")
             if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir()}
    runs = [(s, base_id if s == 0 else ckpts[s]) for s in args.steps if s == 0 or s in ckpts]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.jsonl"

    print(f"model={args.model}  pairs={len(eng_texts)}  steps={[s for s, _ in runs]}  "
          f"device={device}  dtype={args.dtype}")
    for step, path in runs:
        t0 = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(device)
        encoder = model.get_encoder()
        encoder.eval()
        eng_layers = encode_layers(encoder, tokenizer, eng_texts, args.batch_size, device)
        xho_layers = encode_layers(encoder, tokenizer, xho_texts, args.batch_size, device)

        print(f"\nstep {step}  ({len(eng_layers)} layers, {time.time() - t0:.0f}s to encode)")
        print(f"{'layer':>6} {'paired':>8} {'random':>8} {'margin':>8} {'R@1 e>x':>8} {'R@1 x>e':>8}")
        with open(out_path, "a") as f:
            for layer, (e, x) in enumerate(zip(eng_layers, xho_layers)):
                stats = alignment_stats(e, x)
                print(f"{layer:>6} {stats['paired_cos']:>8.4f} {stats['random_cos']:>8.4f} "
                      f"{stats['margin']:>8.4f} {stats['retrieval_e2x']:>8.2%} "
                      f"{stats['retrieval_x2e']:>8.2%}")
                f.write(json.dumps({"model": args.model, "step": step, "layer": layer,
                                    **stats, "n_pairs": len(eng_texts),
                                    "dtype": args.dtype}) + "\n")
        del model, encoder, eng_layers, xho_layers
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nrows appended to {out_path}")


if __name__ == "__main__":
    main()

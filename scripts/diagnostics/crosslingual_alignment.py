"""
Cross-lingual alignment of encoder representations across CPT checkpoints.

For each checkpoint (step 0 = un-adapted model, then the 19 CPT
snapshots): encode the 997 FLORES dev English sentences and their exact
isiXhosa translations with the model's ENCODER, mean-pool the final
hidden states over non-padding positions, and measure how aligned the two
languages' representation spaces are:

Metrics and names follow Idris et al. ("Can Embedding Similarity Predict
Cross-Lingual Transfer? A Systematic Study on African Languages"):
  cosine_mean   mean cos(eng_i, xho_i) over aligned (translation) pairs
  baseline      mean over ALL N^2 entries of the similarity matrix -
                the anisotropy floor (Eq. 1's second term)
  cosine_gap    cosine_mean - baseline (Eq. 1): the cross-lingual signal,
                corrected for "everything is close to everything"
  P@1           % of sentences whose nearest neighbour in the other
                language is the true translation; asymmetric, both ways

Legacy aliases (paired_cos / random_cos / margin) are still written so
rows stay readable by anything built before the rename.

Plotted against CPT step this shows whether isiXhosa-only CPT pulls the
two languages together (byt5 learning isiXhosa) or apart (nguni drifting
from English - the proposed mechanism for its MT en->xh decline).

Usage:
    uv run python3 -m scripts.diagnostics.crosslingual_alignment --model byt5
    uv run python3 -m scripts.diagnostics.crosslingual_alignment --model t5 \
        --steps 0 1000 10000 --limit 100
"""

import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from datasets import load_from_disk

MODELS = {
    "t5": ("google-t5/t5-large", "/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints"),
    "byt5": ("google/byt5-large", "/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints"),
}
FLORES_DIR = "data/finetune/mt"
SPLIT = "devtest"   # matches Idris et al. exactly (1,012 sentences); the
                    # probe is a frozen forward pass - nothing trains on it


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Cross-lingual encoder alignment per checkpoint.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS))
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out-dir", type=str, default="results/alignment")
    parser.add_argument("--base-override", type=str, default=None)
    parser.add_argument("--checkpoints-override", type=str, default=None)
    parser.add_argument("--data-override", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def encode(encoder, tokenizer, texts: list[str], batch_size: int, device) -> torch.Tensor:
    """Mean-pooled final hidden states, L2-normalised, fp32."""
    out = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=512).to(device)
        hidden = encoder(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(out)


def alignment_stats(eng: torch.Tensor, xho: torch.Tensor) -> dict:
    """Verbatim Idris et al.: cosine_mean = mean of the diagonal of M;
    cosine_gap (Eq. 1) = cosine_mean - mean over ALL N^2 entries of M
    (baseline includes the diagonal); P@1 = fraction whose nearest
    neighbour is the true translation, both directions."""
    sims = eng @ xho.T                      # M: (n, n) cosine matrix
    n = sims.shape[0]
    cosine_mean = sims.diagonal().mean().item()
    baseline = sims.mean().item()           # (1/N^2) * sum_ij M_ij
    gap = cosine_mean - baseline
    r_e2x = (sims.argmax(dim=1) == torch.arange(n)).float().mean().item()
    r_x2e = (sims.argmax(dim=0) == torch.arange(n)).float().mean().item()
    return {"cosine_mean": cosine_mean, "baseline": baseline, "cosine_gap": gap,
            "p_at_1_e2x": r_e2x, "p_at_1_x2e": r_x2e,
            # legacy aliases, pre-rename readers
            "paired_cos": cosine_mean, "random_cos": baseline, "margin": gap,
            "retrieval_e2x": r_e2x, "retrieval_x2e": r_x2e}


def main() -> None:
    args = parse_args()
    base_id, ckpt_root = MODELS[args.model]
    if args.base_override:
        base_id = args.base_override
    root = Path(args.checkpoints_override or ckpt_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    data_path = args.data_override or FLORES_DIR
    if str(data_path).endswith(".jsonl"):
        # plain {"source","target"} lines - smoke tests / non-datasets envs
        rows = [json.loads(l) for l in Path(data_path).read_text().splitlines() if l.strip()]
        eng_texts = [r["source"].strip() for r in rows]
        xho_texts = [r["target"].strip() for r in rows]
    else:
        rows_ds = load_from_disk(data_path)[SPLIT]
        eng_texts = [s.strip() for s in rows_ds["source"]]
        xho_texts = [t.strip() for t in rows_ds["target"]]
    if args.limit:
        eng_texts, xho_texts = eng_texts[:args.limit], xho_texts[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    ckpts = sorted((int(m.group(1)), str(p)) for p in root.glob("checkpoint-*")
                   if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir())
    runs = [(0, base_id)] + ckpts
    if args.steps is not None:
        runs = [(s, p) for s, p in runs if s in set(args.steps)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.jsonl"

    print(f"model={args.model}  pairs={len(eng_texts)}  device={device}  dtype={args.dtype}")
    print(f"{'step':>7} {'cos_mean':>9} {'baseline':>9} {'cos_gap':>8} {'P@1 e>x':>8} {'P@1 x>e':>8} {'sec':>5}")

    results = []
    for step, path in runs:
        t0 = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(device)
        encoder = model.get_encoder()
        encoder.eval()
        eng = encode(encoder, tokenizer, eng_texts, args.batch_size, device)
        xho = encode(encoder, tokenizer, xho_texts, args.batch_size, device)
        stats = alignment_stats(eng, xho)
        results.append((step, stats["cosine_gap"]))
        print(f"{step:>7} {stats['cosine_mean']:>9.4f} {stats['baseline']:>9.4f} "
              f"{stats['cosine_gap']:>8.4f} {stats['p_at_1_e2x']:>8.2%} "
              f"{stats['p_at_1_x2e']:>8.2%} {time.time() - t0:>5.0f}")
        with open(out_path, "a") as f:
            f.write(json.dumps({"model": args.model, "step": step, **stats,
                                "n_pairs": len(eng_texts), "dtype": args.dtype}) + "\n")
        del model, encoder
        if device == "cuda":
            torch.cuda.empty_cache()

    cpt = [(s, m) for s, m in results if s > 0]
    if len(cpt) >= 3:
        import numpy as np
        steps_, gaps_ = zip(*cpt)
        ra = np.argsort(np.argsort(steps_)).astype(float)
        rb = np.argsort(np.argsort(gaps_)).astype(float)
        rho = float(np.corrcoef(ra, rb)[0, 1])
        print(f"\nrho(step, cosine_gap): {rho:+.3f}   "
              f"(positive = CPT aligns eng/xho spaces, negative = drives them apart)")
    print(f"rows appended to {out_path}")


if __name__ == "__main__":
    main()

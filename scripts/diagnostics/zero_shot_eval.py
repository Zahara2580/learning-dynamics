"""
Zero-shot (NO finetuning) evaluation of base + every CPT checkpoint on
the SAME test set, references, and frozen generation settings as the
finetuned evaluation - so the rows are directly comparable.

Per checkpoint:
  - teacher-forced loss on the test pairs (cheap, one forward sweep)
  - beam-search generation with the task config's pinned settings
  - BLEU / chrF / chrF++ / TER via the same score_corpus as the harness
  - every prediction stored, in two forms: "pred" (specials stripped,
    exactly what gets scored) and "raw" (sentinels/pads visible, for
    eyeballing what the denoising model actually emits)

Expectation to read the numbers against: span-corruption models given
task-formatted input produce denoising-style output, so absolute scores
will be far below finetuned ones for ALL checkpoints. The evidence is in
the DIFFERENCES across checkpoints and in whether the raw text drifts
toward isiXhosa as CPT progresses.

Usage:
    uv run python3 -m scripts.diagnostics.zero_shot_eval --model byt5 --task d2t
    uv run python3 -m scripts.diagnostics.zero_shot_eval --model t5 --task d2t \
        --steps 0 1000 10000 --limit 50        # CPU-sized spot check
"""

import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from src.finetuning.config import FinetuneConfig
from src.finetuning.data_mt import FLORES_TEST_SPLIT, build_mt_sources, load_flores_split
from src.finetuning.data_t2x import build_training_pairs, load_t2x_split
from src.finetuning.metrics import score_corpus
from src.finetuning.run_finetune import Seq2SeqDataset

MODELS = {
    "t5": ("google-t5/t5-large", "/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints"),
    "byt5": ("google/byt5-large", "/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints"),
}
CONFIGS = {
    "d2t": "configs/finetune/d2t.yaml",
    "mt": "configs/finetune/mt.yaml",
    "mt-xhen": "configs/finetune/mt_xhen_5epoch.yaml",
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot metrics across CPT checkpoints.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS))
    parser.add_argument("--task", type=str, default="d2t", choices=list(CONFIGS))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Subset of checkpoint steps (0 = base). Default: base + all.")
    parser.add_argument("--limit", type=int, default=None, help="Cap test examples (spot checks).")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out-dir", type=str, default="results/zero_shot")
    parser.add_argument("--base-override", type=str, default=None)
    parser.add_argument("--checkpoints-override", type=str, default=None)
    parser.add_argument("--data-override", type=str, default=None)
    return parser.parse_args()


def load_eval_data(task: str, data_dir: str, cfg: FinetuneConfig, limit: int | None):
    """Test sources, loss targets (first ref), and full scoring references -
    identical to what the finetuned evaluation uses."""
    if task == "d2t":
        inputs, refs = load_t2x_split(Path(data_dir), "test")
        sources, loss_targets = build_training_pairs(inputs, refs, cfg.source_prefix)
    else:
        raw, refs = load_flores_split(data_dir, FLORES_TEST_SPLIT, cfg.direction)
        sources = build_mt_sources(raw, cfg.source_prefix, cfg.direction_prefix)
        loss_targets = [r[0] for r in refs]
    if limit:
        sources, loss_targets, refs = sources[:limit], loss_targets[:limit], refs[:limit]
    return sources, loss_targets, refs


@torch.no_grad()
def token_weighted_loss(model, loader, device) -> float:
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def generate_both(model, tokenizer, sources: list[str], cfg: FinetuneConfig, device):
    """Frozen-settings generation; returns (scored_text, raw_text) per source."""
    preds, raws = [], []
    for start in range(0, len(sources), cfg.eval_batch_size):
        batch = sources[start:start + cfg.eval_batch_size]
        enc = tokenizer(batch, max_length=cfg.max_source_length, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            num_beams=cfg.num_beams,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=False,
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
        )
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        raws.extend(tokenizer.batch_decode(gen, skip_special_tokens=False))
    return preds, raws


def main() -> None:
    args = parse_args()
    base_id, ckpt_root = MODELS[args.model]
    if args.base_override:
        base_id = args.base_override
    root = Path(args.checkpoints_override or ckpt_root)
    cfg = FinetuneConfig.from_yaml(CONFIGS[args.task])
    data_dir = args.data_override or cfg.data_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    sources, loss_targets, references = load_eval_data(args.task, data_dir, cfg, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    dataset = Seq2SeqDataset(sources, loss_targets, tokenizer,
                             cfg.max_source_length, cfg.max_target_length)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size,
                        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=None))

    ckpts = sorted((int(m.group(1)), str(p)) for p in root.glob("checkpoint-*")
                   if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir())
    runs = [(0, base_id)] + ckpts
    if args.steps is not None:
        runs = [(s, p) for s, p in runs if s in set(args.steps)]

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.task}.jsonl"

    print(f"model={args.model}  task={args.task}  n={len(sources)}  device={device}  "
          f"dtype={args.dtype}  beams={cfg.num_beams}  max_new={cfg.max_new_tokens}")
    print(f"{'step':>7} {'loss':>9} {'chrF':>7} {'BLEU':>7} {'chrF++':>7} {'TER':>8} {'sec':>6}")

    base_chrf, rows = None, []
    for step, path in runs:
        t0 = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(device)
        model.eval()

        loss = token_weighted_loss(model, loader, device)
        preds, raws = generate_both(model, tokenizer, sources, cfg, device)
        scored = score_corpus(preds, references)
        m = scored["metrics"]
        if base_chrf is None:
            base_chrf = m["chrf"]
        rows.append((step, m["chrf"]))

        print(f"{step:>7} {loss:>9.4f} {m['chrf']:>7.2f} {m['bleu']:>7.2f} "
              f"{m['chrf_pp']:>7.2f} {m['ter']:>8.2f} {time.time() - t0:>6.0f}")
        for r in raws[:2]:
            print(f"          raw>> {r[:140]!r}")

        with open(pred_dir / f"{args.model}_{step}_{args.task}.jsonl", "w", encoding="utf-8") as f:
            for i, (p, r) in enumerate(zip(preds, raws)):
                f.write(json.dumps({"i": i, "pred": p, "raw": r}, ensure_ascii=False) + "\n")

        with open(out_path, "a") as f:
            f.write(json.dumps({
                "model": args.model, "task": args.task, "step": step,
                "zero_shot_loss": loss, "metrics": m,
                "diagnostics": scored["diagnostics"],
                "n_examples": len(sources), "dtype": args.dtype,
            }) + "\n")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    cpt = [(s, c) for s, c in rows if s > 0]
    if len(cpt) >= 3:
        import numpy as np
        steps_, chrfs_ = zip(*cpt)
        ra = np.argsort(np.argsort(steps_)).astype(float)
        rb = np.argsort(np.argsort(chrfs_)).astype(float)
        rho = float(np.corrcoef(ra, rb)[0, 1])
        print(f"\nrho(step, zero-shot chrF): {rho:+.3f}   base chrF {base_chrf:.2f}")
    print(f"summary -> {out_path}\npredictions -> {pred_dir}/")


if __name__ == "__main__":
    main()

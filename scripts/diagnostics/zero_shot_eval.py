"""
Zero-shot (NO finetuning) evaluation of base + every CPT checkpoint.

Measures teacher-forced cross-entropy on the downstream pairs exactly as
the finetuning harness would see them, but with raw checkpoint weights.
If CPT taught the model isiXhosa, the loss on isiXhosa targets must fall
with CPT step for t5/byt5 (and stay near-flat for the already-adapted
nguni-byt5). If it is flat for byt5/t5, CPT did not transfer - either
way, hard evidence that the checkpoints differ behaviourally BEFORE any
finetuning touches them.

Loss is aggregated token-weighted over the split (identical semantics to
the Trainer's eval_loss). Optional --generate prints a few greedy
continuations per checkpoint for qualitative eyeballing - expect
denoising-style output with sentinels, NOT fluent task output; the
information is in whether it looks like isiXhosa.

Usage:
    uv run python3 -m scripts.diagnostics.zero_shot_eval --model byt5 --task d2t
    uv run python3 -m scripts.diagnostics.zero_shot_eval --model t5 --task d2t \
        --steps 0 1000 10000 --limit 100          # CPU-sized spot check
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

from src.finetuning.data_mt import FLORES_TEST_SPLIT, build_mt_sources, load_flores_split
from src.finetuning.data_t2x import build_training_pairs, load_t2x_split
from src.finetuning.run_finetune import Seq2SeqDataset

MODELS = {
    "t5": ("google-t5/t5-large", "/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints"),
    "byt5": ("google/byt5-large", "/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints"),
}
DATA = {"d2t": "data/finetune/d2t", "mt": "data/finetune/mt"}
MT_PREFIX = "Translate English to Xhosa: "


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot loss across CPT checkpoints.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS))
    parser.add_argument("--task", type=str, default="d2t", choices=["d2t", "mt"])
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Subset of checkpoint steps (0 = base). Default: base + all.")
    parser.add_argument("--limit", type=int, default=None, help="Cap test examples (CPU spot checks).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--generate", type=int, default=0,
                        help="Print N greedy sample continuations per checkpoint.")
    parser.add_argument("--out-dir", type=str, default="results/zero_shot")
    parser.add_argument("--base-override", type=str, default=None)
    parser.add_argument("--checkpoints-override", type=str, default=None)
    parser.add_argument("--data-override", type=str, default=None)
    return parser.parse_args()


def load_pairs(task: str, data_dir: str, limit: int | None) -> tuple[list[str], list[str]]:
    if task == "d2t":
        inputs, refs = load_t2x_split(Path(data_dir), "test")
        sources, targets = build_training_pairs(inputs, refs, "")
    else:
        raw, refs = load_flores_split(data_dir, FLORES_TEST_SPLIT)
        sources = build_mt_sources(raw, "", MT_PREFIX)
        targets = [r[0] for r in refs]
    if limit:
        sources, targets = sources[:limit], targets[:limit]
    return sources, targets


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


def main() -> None:
    args = parse_args()
    base_id, ckpt_root = MODELS[args.model]
    if args.base_override:
        base_id = args.base_override
    root = Path(args.checkpoints_override or ckpt_root)
    data_dir = args.data_override or DATA[args.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    sources, targets = load_pairs(args.task, data_dir, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    dataset = Seq2SeqDataset(sources, targets, tokenizer, 512, 512)
    collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    ckpts = sorted((int(m.group(1)), str(p)) for p in root.glob("checkpoint-*")
                   if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir())
    runs = [(0, base_id)] + ckpts
    if args.steps is not None:
        runs = [(s, p) for s, p in runs if s in set(args.steps)]

    print(f"model={args.model}  task={args.task}  n={len(dataset)}  device={device}  dtype={args.dtype}")
    print(f"{'step':>7} {'zero-shot loss':>15} {'d vs base':>10} {'seconds':>8}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.task}.jsonl"

    base_loss, rows = None, []
    for step, path in runs:
        t0 = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=dtype).to(device)
        model.eval()
        loss = token_weighted_loss(model, loader, device)
        if base_loss is None:
            base_loss = loss
        rows.append((step, loss))
        print(f"{step:>7} {loss:>15.4f} {loss - base_loss:>+10.4f} {time.time() - t0:>8.0f}")

        if args.generate:
            enc = tokenizer(sources[:args.generate], return_tensors="pt",
                            padding=True, truncation=True, max_length=512).to(device)
            gen = model.generate(**enc, max_new_tokens=48, num_beams=1, do_sample=False)
            for g in tokenizer.batch_decode(gen, skip_special_tokens=False):
                print(f"          >> {g[:150]!r}")

        with open(out_path, "a") as f:
            f.write(json.dumps({"model": args.model, "task": args.task, "step": step,
                                "zero_shot_loss": loss, "n_examples": len(dataset),
                                "dtype": args.dtype}) + "\n")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    cpt = [(s, l) for s, l in rows if s > 0]
    if len(cpt) >= 3:
        import numpy as np
        steps_, losses_ = zip(*cpt)
        ra = np.argsort(np.argsort(steps_)).astype(float)
        rb = np.argsort(np.argsort(losses_)).astype(float)
        rho = float(np.corrcoef(ra, rb)[0, 1])
        print(f"\nrho(step, zero-shot loss) over CPT checkpoints: {rho:+.3f}  "
              f"(negative = CPT reduces loss on {args.task} targets)")
    print(f"rows appended to {out_path}")


if __name__ == "__main__":
    main()

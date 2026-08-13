"""
T2X extractive slot metrics (Meyer et al.): subject/object copy-vs-
translate P/R/F1 over saved D2T predictions.

Per triple: GOLD = "copy" if the English entity string occurs in a
reference text (else "translate"); PRED = whether it occurs in the
generation. P/R/F1 with copy as the positive class, subjects (entity
tags) and objects (value tags) separately. Relation prediction is NOT
implemented (needs a trained classifier).

Defaults where the paper is silent (both flaggable):
  --gold-ref any    entity in ANY of the 1-8 references counts as copy
  (occurrences)     duplicate entity strings within an example count once

CPU-only, post-hoc over results*/predictions/*_d2t_*; resumable.

Usage:
    uv run python3 -m scripts.diagnostics.t2x_slot_metrics
"""

import argparse
import json
import re
from argparse import Namespace
from pathlib import Path

from src.finetuning.data_t2x import load_t2x_split

ARMS = [
    ("results/finetune/predictions", "d2t-main"),
    ("results_d2t_warmup_20/predictions", "d2t-warmup20"),
]
ENTITY_RE = re.compile(r"__start_entity__ (.*?) __end_entity__")
VALUE_RE = re.compile(r"__start_value__ (.*?) __end_value__")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="T2X subject/object copy F1.")
    parser.add_argument("--data-dir", type=str, default="data/finetune/d2t")
    parser.add_argument("--out", type=str, default="results/t2x_slots/scores.jsonl")
    parser.add_argument("--gold-ref", type=str, default="any", choices=["any", "first"])
    parser.add_argument("--per-triple", action="store_true",
                        help="Count every triple occurrence (default: unique per example).")
    parser.add_argument("--lowercase", action="store_true",
                        help="Case-insensitive matching (default: case-sensitive).")
    return parser.parse_args()


def parse_name(stem: str) -> dict | None:
    p = stem.split("_")
    if len(p) == 6:
        return {"model": p[0], "step": int(p[1]), "task": p[2], "seed": int(p[3]),
                "selection": f"{p[4]}_{p[5]}"}
    if len(p) == 4:
        return {"model": p[0], "step": int(p[1]), "task": p[2], "seed": int(p[3]),
                "selection": None}
    return None


def norm(text: str, lowercase: bool) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def f1_counts(decisions: list[tuple[bool, bool]]) -> dict:
    """decisions: (gold_copy, pred_copy) per slot; copy = positive class."""
    tp = sum(1 for g, p in decisions if g and p)
    fp = sum(1 for g, p in decisions if not g and p)
    fn = sum(1 for g, p in decisions if g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"p": round(100 * precision, 2), "r": round(100 * recall, 2),
            "f1": round(100 * f1, 2), "n": len(decisions),
            "gold_copy_rate": round(sum(g for g, _ in decisions) / max(1, len(decisions)), 4)}


def main() -> None:
    args = parse_args()
    inputs, references = load_t2x_split(Path(args.data_dir), "test")

    # per example: the slot strings and their gold copy decision
    examples = []
    for inp, refs in zip(inputs, references):
        refs_n = [norm(r, args.lowercase) for r in (refs if args.gold_ref == "any" else refs[:1])]
        slots = {"subject": ENTITY_RE.findall(inp), "object": VALUE_RE.findall(inp)}
        entry = {}
        for role, ents in slots.items():
            if not args.per_triple:
                ents = list(dict.fromkeys(ents))
            entry[role] = [(norm(e, args.lowercase),
                            any(norm(e, args.lowercase) in r for r in refs_n)) for e in ents]
        examples.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["file"]))

    settings = {"gold_ref": args.gold_ref, "per_triple": args.per_triple,
                "lowercase": args.lowercase}
    print(f"test examples: {len(examples)}  settings: {settings}")
    print(f"{'arm':>14} {'model':>11} {'step':>6} {'sel':>11} "
          f"{'subj F1':>8} {'obj F1':>8}")

    for pred_dir, arm in ARMS:
        directory = Path(pred_dir)
        if not directory.is_dir():
            continue
        for f in sorted(directory.glob("*.jsonl")):
            meta = parse_name(f.stem)
            if meta is None or meta["task"] != "d2t" or (arm, f.name) in done:
                continue
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows.sort(key=lambda r: r["i"])
            preds = [norm(r["pred"], args.lowercase) for r in rows]
            if len(preds) != len(examples):
                print(f"  WARNING {f.name}: {len(preds)} preds vs {len(examples)} examples - skipped")
                continue

            decisions = {"subject": [], "object": []}
            for pred, entry in zip(preds, examples):
                for role in decisions:
                    for ent, gold_copy in entry[role]:
                        decisions[role].append((gold_copy, ent in pred))

            row = {"arm": arm, "file": f.name, **meta, **settings,
                   "subject": f1_counts(decisions["subject"]),
                   "object": f1_counts(decisions["object"])}
            with open(out_path, "a") as g:
                g.write(json.dumps(row) + "\n")
            print(f"{arm:>14} {meta['model']:>11} {meta['step']:>6} "
                  f"{str(meta['selection']):>11} {row['subject']['f1']:>8.2f} "
                  f"{row['object']['f1']:>8.2f}")

    print(f"rows -> {out_path}")


if __name__ == "__main__":
    main()

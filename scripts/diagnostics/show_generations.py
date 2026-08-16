"""
Side-by-side generations across CPT checkpoints: the SAME test examples
at every step, so a model's progression on one input is readable in one
block. Works for finetuned sweeps and for zero-shot predictions.

Usage:
    # finetuned D2T, full data
    uv run python3 -m scripts.diagnostics.show_generations \
        --config configs/finetune/d2t_warmup20.yaml \
        --pred-dir results_d2t_warmup_20/predictions --selection last_epoch

    # finetuned MT, full 50k
    uv run python3 -m scripts.diagnostics.show_generations \
        --config configs/finetune/mt_5epoch.yaml \
        --pred-dir results_mt_5epoch/predictions --selection last_epoch

    # zero-shot (never finetuned); --show-raw exposes sentinels/specials
    uv run python3 -m scripts.diagnostics.show_generations \
        --config configs/finetune/d2t.yaml \
        --pred-dir results/zero_shot/predictions --show-raw
"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

from src.finetuning.config import FinetuneConfig
from src.finetuning.data_mt import FLORES_TEST_SPLIT, load_flores_split
from src.finetuning.data_t2x import load_t2x_split

MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Show generations across checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="Finetune config: supplies data_dir, task, direction.")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory of saved prediction JSONLs.")
    parser.add_argument("--task", type=str, default=None,
                        help="Override the task in filenames (mt-xhen for reverse zero-shot).")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1000, 5000, 10000])
    parser.add_argument("--examples", type=int, nargs="+", default=[0, 1, 2],
                        help="Test-set indices; the same ones are shown for every step.")
    parser.add_argument("--selection", type=str, default=None,
                        help="best_epoch / last_epoch (finetuned files only).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--show-raw", action="store_true",
                        help="Also print the raw decode (specials visible); zero-shot only.")
    return parser.parse_args()


def load_examples(cfg: FinetuneConfig) -> tuple[list[str], list[list[str]]]:
    """Test sources (unprefixed) and their references."""
    if cfg.task == "d2t":
        return load_t2x_split(Path(cfg.data_dir), "test")
    return load_flores_split(cfg.data_dir, FLORES_TEST_SPLIT, cfg.direction)


def find_pred_file(pred_dir: Path, model: str, step: int, task: str,
                   seed: int, selection: str | None) -> Path | None:
    """Match {model}_{step}_{task}[_{seed}[_{selection}]].jsonl exactly."""
    for f in sorted(pred_dir.glob(f"{model}_{step}_*.jsonl")):
        parts = f.stem.split("_")
        if len(parts) < 3 or parts[0] != model or parts[1] != str(step) or parts[2] != task:
            continue
        if len(parts) == 3:                      # zero-shot
            return f
        if len(parts) >= 4 and parts[3] != str(seed):
            continue
        if len(parts) == 6:                      # finetuned, two selections
            if selection and f"{parts[4]}_{parts[5]}" != selection:
                continue
        return f
    return None


def read_rows(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            rows[r["i"]] = r
    return rows


def clip(text: str, limit: int) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + " …"


def main() -> None:
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)
    task = args.task or cfg.task
    pred_dir = Path(args.pred_dir)

    sources, references = load_examples(cfg)
    prefix = (cfg.source_prefix or "") + (cfg.direction_prefix or "")

    # {(model, step): {i: row}}
    loaded, missing = {}, []
    for model in args.models:
        for step in args.steps:
            f = find_pred_file(pred_dir, model, step, task, args.seed, args.selection)
            if f is None:
                missing.append(f"{model}@{step}")
            else:
                loaded[(model, step)] = read_rows(f)

    print(f"task={task}  dir={pred_dir}  selection={args.selection or 'n/a'}")
    print(f"model input prefix: {prefix!r}" if prefix else "model input prefix: (none)")
    if missing:
        print(f"missing predictions: {', '.join(missing)}")

    for i in args.examples:
        if i >= len(sources):
            continue
        print("\n" + "#" * 100)
        print(f"# TEST EXAMPLE {i}")
        print("#" * 100)
        print(f"INPUT      : {clip(sources[i], args.max_chars)}")
        for k, ref in enumerate(references[i][:3]):
            label = "REFERENCE  " if k == 0 else f"  (ref {k + 1})"
            print(f"{label}: {clip(ref, args.max_chars)}")
        if len(references[i]) > 3:
            print(f"             (+{len(references[i]) - 3} more references)")

        for model in args.models:
            steps_here = [s for s in args.steps if (model, s) in loaded]
            if not steps_here:
                continue
            print()
            for step in steps_here:
                row = loaded[(model, step)].get(i)
                if row is None:
                    continue
                tag = "base " if step == 0 else f"{step:>5}"
                print(f"  {model:<11} step {tag} : {clip(row['pred'], args.max_chars)}")
                if args.show_raw and "raw" in row:
                    print(f"  {'':<11}      raw   : {clip(row['raw'], args.max_chars)}")


if __name__ == "__main__":
    main()

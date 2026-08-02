"""
Summarise the finetuning sweep: per-checkpoint metrics, deltas vs base,
trend, and sanity audits.

Usage:
    uv run python3 -m src.finetuning.summarize_results
    uv run python3 -m src.finetuning.summarize_results --metric chrf
"""

import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Summarise finetuning results.")
    parser.add_argument("--results", type=str, default="results/finetune/results.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="chrf", help="Metric for the trend test.")
    return parser.parse_args()


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation, without a scipy dependency."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["seed"] == args.seed]
    if not rows:
        raise SystemExit(f"no rows with seed {args.seed} in {args.results}")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["task"], r.get("selection", "best_epoch"))].append(r)

    print(f"{len(rows)} rows across {len(groups)} (model, task) groups\n")

    for (model, task, selection), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["ckpt_step"])
        base = next((r for r in rs if r["ckpt_step"] == 0), None)

        print("=" * 96)
        print(f"{model}  /  {task}  /  {selection}      {len(rs)}/20 checkpoints")
        print("=" * 96)
        print(f"{'step':>7} {'val_loss':>9} {'chrF':>7} {'BLEU':>7} {'chrF++':>7} {'TER':>7} "
              f"{'dchrF':>7} {'dBLEU':>7} {'ep':>3} {'empty%':>7} {'rep%':>6} {'len_r':>6}")

        for r in rs:
            m, d = r["metrics"], r["diagnostics"]
            val = min(r["val_loss_per_epoch"]) if r["val_loss_per_epoch"] else float("nan")
            dchrf = m["chrf"] - base["metrics"]["chrf"] if base else float("nan")
            dbleu = m["bleu"] - base["metrics"]["bleu"] if base else float("nan")
            n = max(1, d.get("n_preds", 1))
            star = "  <- base" if r["ckpt_step"] == 0 else ""
            print(f"{r['ckpt_step']:>7} {val:>9.4f} {m['chrf']:>7.2f} {m['bleu']:>7.2f} "
                  f"{m['chrf_pp']:>7.2f} {m['ter']:>7.2f} {dchrf:>+7.2f} {dbleu:>+7.2f} "
                  f"{r['best_epoch'] or 0:>3} {100 * d['n_empty'] / n:>7.2f} "
                  f"{100 * d['n_repetitive'] / n:>6.2f} {d['pred_ref_len_ratio']:>6.2f}{star}")

        # Trend: does the metric move with CPT step (excluding base)?
        cpt = [r for r in rs if r["ckpt_step"] > 0]
        if len(cpt) >= 3:
            steps = [r["ckpt_step"] for r in cpt]
            vals = [r["metrics"][args.metric] for r in cpt]
            losses = [min(r["val_loss_per_epoch"]) for r in cpt if r["val_loss_per_epoch"]]
            rho = spearman(steps, vals)
            print(f"\n  trend over CPT checkpoints (excl. base):")
            print(f"    {args.metric} vs step : rho={rho:+.3f}   "
                  f"range {min(vals):.2f}..{max(vals):.2f} (spread {max(vals) - min(vals):.2f})")
            if len(losses) == len(steps):
                print(f"    val_loss vs step : rho={spearman(steps, losses):+.3f}   "
                      f"range {min(losses):.4f}..{max(losses):.4f}")
            if base:
                better = sum(1 for r in cpt if r["metrics"][args.metric] > base["metrics"][args.metric])
                print(f"    beat base on {args.metric}: {better}/{len(cpt)} checkpoints")
                bl = min(base["val_loss_per_epoch"])
                better_l = sum(1 for r in cpt if min(r["val_loss_per_epoch"]) < bl)
                print(f"    beat base on val_loss  : {better_l}/{len(cpt)} checkpoints")

        # Audit: decoding settings must be identical across checkpoints.
        defaults = {json.dumps(r.get("generation", {}).get("checkpoint_defaults", {}), sort_keys=True)
                    for r in rs}
        print(f"\n  generation defaults across checkpoints: "
              f"{'IDENTICAL' if len(defaults) == 1 else f'*** {len(defaults)} VARIANTS - asymmetric decoding ***'}")
        hashes = {r["config_hash"] for r in rs}
        print(f"  config_hash: {'single protocol' if len(hashes) == 1 else f'*** {len(hashes)} DIFFERENT PROTOCOLS: {hashes} ***'}")
        # Rates, not a boolean: "any repetitive prediction" fires on every MT
        # checkpoint (1012 test sentences) and so carries no information.
        rates = [(r["ckpt_step"],
                  100 * (r["diagnostics"]["n_empty"] + r["diagnostics"]["n_repetitive"])
                  / max(1, r["diagnostics"].get("n_preds", 1))) for r in rs]
        worst = max(rates, key=lambda x: x[1])
        mean_rate = sum(v for _, v in rates) / len(rates)
        print(f"  degenerate output: mean {mean_rate:.2f}% of predictions, "
              f"worst {worst[1]:.2f}% at step {worst[0]}")
        print()


if __name__ == "__main__":
    main()

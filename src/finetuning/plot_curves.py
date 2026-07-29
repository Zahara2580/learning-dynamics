"""
Learning-dynamics curves: one PNG per (task, metric).

Base (step 0) is drawn as a dashed horizontal reference, not a point on
the curve - it is the un-adapted starting model, not a CPT checkpoint.
Log x-axis so the dense 100-1000 checkpoints are readable.

Usage:
    uv run python3 -m src.finetuning.plot_curves
    uv run python3 -m src.finetuning.plot_curves --results r.jsonl --output-dir plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["bleu", "chrf", "chrf_pp", "ter"]
LABELS = {"bleu": "BLEU", "chrf": "chrF", "chrf_pp": "chrF++", "ter": "TER"}
# Headline metric per task, per the proposal.
HEADLINE = {"d2t": "chrf", "mt": "bleu"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning-dynamics curves.")
    parser.add_argument("--results", type=str, default="results/finetune/results.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/finetune/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["seed"] == args.seed]
    if not rows:
        raise SystemExit(f"no rows with seed {args.seed} in {args.results}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            if metric in row["metrics"]:
                series[(row["task"], metric)][row["model"]].append(
                    (row["ckpt_step"], row["metrics"][metric]))

    for (task, metric), by_model in sorted(series.items()):
        figure, axis = plt.subplots(figsize=(8, 5))
        for i, (model, points) in enumerate(sorted(by_model.items())):
            points.sort()
            colour = f"C{i}"
            base = [v for s, v in points if s == 0]
            cpt = [(s, v) for s, v in points if s > 0]
            if base:
                axis.axhline(base[0], color=colour, linestyle="--", linewidth=1.2, alpha=0.8)
                axis.text(cpt[-1][0] if cpt else 1, base[0], f" {model} base",
                          color=colour, va="bottom", ha="right", fontsize=8)
            if cpt:
                axis.plot([s for s, _ in cpt], [v for _, v in cpt],
                          marker="o", markersize=4, color=colour, label=model)

        axis.set_xscale("log")
        axis.set_xlabel("CPT step (log scale); dashed line = un-adapted base model")
        axis.set_ylabel(LABELS[metric])
        star = "  [headline metric]" if HEADLINE.get(task) == metric else ""
        axis.set_title(f"{task.upper()}: {LABELS[metric]} vs CPT step{star}")
        axis.grid(alpha=0.3, which="both")
        axis.legend(loc="best", fontsize=9)
        figure.tight_layout()

        path = output_dir / f"{task}_{metric}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

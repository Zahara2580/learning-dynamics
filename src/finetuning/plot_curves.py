"""
Plot learning-dynamics curves from the finetuning results.

One PNG per (task, metric): x = CPT step, one line per model. Minimal by
design - this exists to eyeball the curves during the sweep, not to
produce thesis figures.

Usage:
    uv run python3 -m src.finetuning.plot_curves \
        --results results/finetune/results.jsonl --output-dir results/finetune/plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["bleu", "chrf", "chrf_pp", "ter"]


def main() -> None:
    """Read results.jsonl and write one PNG per (task, metric)."""
    parser = argparse.ArgumentParser(description="Plot learning-dynamics curves.")
    parser.add_argument("--results", type=str, default="results/finetune/results.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/finetune/plots")
    parser.add_argument("--seed", type=int, default=42, help="Only plot rows with this seed.")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.results).read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["seed"] == args.seed]
    if not rows:
        raise SystemExit(f"no rows with seed {args.seed} in {args.results}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (task, metric) -> model -> [(step, score), ...]
    series = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            if metric in row["metrics"]:
                series[(row["task"], metric)][row["model"]].append(
                    (row["ckpt_step"], row["metrics"][metric])
                )

    for (task, metric), by_model in sorted(series.items()):
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for model, points in sorted(by_model.items()):
            points.sort()
            axis.plot([p[0] for p in points], [p[1] for p in points],
                      marker="o", markersize=4, label=model)
        axis.set_xlabel("CPT step (0 = un-adapted base model)")
        axis.set_ylabel(metric.replace("_pp", "++").upper())
        axis.set_title(f"{task}: {metric.replace('_pp', '++').upper()} vs CPT step")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()

        path = output_dir / f"{task}_{metric}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

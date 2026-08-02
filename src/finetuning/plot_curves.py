"""
Learning-dynamics curves: downstream metric vs CPT step.

Writes one figure per (task, metric, model) on its own y-scale, plus a
combined figure per (task, metric). The per-model figures are the ones to
read for trends - on a shared axis a 1-point trend inside a 25-point range
looks flat.

Base (step 0) is a dashed horizontal reference, not a point on the curve:
it is the un-adapted starting model, not a CPT checkpoint.

Usage:
    uv run python3 -m src.finetuning.plot_curves
    uv run python3 -m src.finetuning.plot_curves --results results/variance/results.jsonl \
        --output-dir results/variance/plots
"""

import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["bleu", "chrf", "chrf_pp", "ter"]
LABELS = {"bleu": "BLEU", "chrf": "chrF", "chrf_pp": "chrF++", "ter": "TER"}
MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Plot learning-dynamics curves.")
    parser.add_argument("--results", type=str, default="results/finetune/results.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/finetune/plots")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def colour(model: str) -> str:
    return f"C{MODEL_ORDER.index(model)}" if model in MODEL_ORDER else "C7"


# best_epoch vs last_epoch share a model's colour; the line style separates them
STYLES = {"best_epoch": "-", "last_epoch": ":"}


def draw(curves: dict[str, list[tuple[int, float]]], title: str, ylabel: str, path: Path) -> None:
    """One figure; one line per model, each with its own base reference."""
    figure, axis = plt.subplots(figsize=(8, 5))
    drawn_base = set()
    for key, points in sorted(curves.items(), key=lambda kv: (
            MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 99, kv[0][1])):
        model, selection = key
        points = sorted(points)
        base = [v for s, v in points if s == 0]
        cpt = [(s, v) for s, v in points if s > 0]
        c = colour(model)
        if base and model not in drawn_base:
            drawn_base.add(model)
            axis.axhline(base[0], color=c, linestyle="--", linewidth=1.2, alpha=0.8)
            if cpt:
                axis.text(cpt[-1][0], base[0], f" {model} base", color=c,
                          va="bottom", ha="right", fontsize=8)
        if cpt:
            label = model if selection is None else f"{model} ({selection})"
            axis.plot([s for s, _ in cpt], [v for _, v in cpt], marker="o", markersize=4,
                      color=c, linestyle=STYLES.get(selection, "-"), label=label)

    axis.set_xscale("log")
    axis.set_xlabel("CPT step")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3, which="both")
    if len(curves) > 1:
        axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["seed"] == args.seed]
    if not rows:
        raise SystemExit(f"no rows with seed {args.seed} in {args.results}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # None when a file predates the selection field, so old results still plot
    selections = {r.get("selection") for r in rows}
    multi = len(selections) > 1
    series = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sel = row.get("selection") if multi else None
        for metric in METRICS:
            if metric in row["metrics"]:
                series[(row["task"], metric)][(row["model"], sel)].append(
                    (row["ckpt_step"], row["metrics"][metric]))

    for (task, metric), by_model in sorted(series.items()):
        label = LABELS[metric]
        # per model, own y-scale - read these for trends
        models = {k[0] for k in by_model}
        for model in sorted(models):
            subset = {k: v for k, v in by_model.items() if k[0] == model}
            draw(subset, f"{task.upper()} {label} for {model}", label,
                 output_dir / f"{task}_{metric}_{model}.png")
        # all models together - for cross-model comparison only
        if len(models) > 1:
            draw(dict(by_model), f"{task.upper()} {label}", label,
                 output_dir / f"{task}_{metric}_all.png")


if __name__ == "__main__":
    main()

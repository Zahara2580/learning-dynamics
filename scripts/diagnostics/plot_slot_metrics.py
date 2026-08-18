"""
T2X slot metrics vs CPT step: subject/object copy F1 (plus the
precision/recall decomposition, which is where the failure mode shows).

One figure per (arm, metric, model) on its own y-scale, plus a combined
figure per (arm, metric). Base (step 0) is a dashed reference line, not a
point on the curve - same conventions as plot_curves.py.

Usage:
    uv run python3 -m scripts.diagnostics.plot_slot_metrics
    uv run python3 -m scripts.diagnostics.plot_slot_metrics \
        --scores results/t2x_slots/scores_firstref.jsonl \
        --output-dir results/t2x_slots/plots_firstref
"""

import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]

# label, role key, stat key
METRICS = {
    "subject_f1": ("Subject F1", "subject", "f1"),
    "object_f1": ("Object F1", "object", "f1"),
    "subject_p": ("Subject precision", "subject", "p"),
    "subject_r": ("Subject recall", "subject", "r"),
    "object_p": ("Object precision", "object", "p"),
    "object_r": ("Object recall", "object", "r"),
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Plot T2X slot metrics per checkpoint.")
    parser.add_argument("--scores", type=str, default="results/t2x_slots/scores.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--selection", type=str, default="last_epoch",
                        help="Rows with no selection field (pre-two-selection arms) always pass.")
    parser.add_argument("--arms", nargs="+", default=None, help="Filter to these arms.")
    return parser.parse_args()


def colour(model: str) -> str:
    return f"C{MODEL_ORDER.index(model)}" if model in MODEL_ORDER else "C7"


def draw(series: dict[str, list[tuple[int, float]]], title: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for model in sorted(series, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
        points = sorted(series[model])
        base = [v for s, v in points if s == 0]
        cpt = [(s, v) for s, v in points if s > 0]
        c = colour(model)
        if base:
            axis.axhline(base[0], color=c, linestyle="--", linewidth=1.0, alpha=0.55)
            if cpt:
                axis.text(cpt[-1][0], base[0], f" {model} base", color=c,
                          va="bottom", ha="right", fontsize=7, alpha=0.9)
        if cpt:
            axis.plot([s for s, _ in cpt], [v for _, v in cpt],
                      marker="o", markersize=4, color=c, label=model)
    axis.set_xlabel("CPT step")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    if len(series) > 1:
        axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    scores = Path(args.scores)
    rows = [json.loads(l) for l in scores.read_text(encoding="utf-8").split("\n") if l.strip()]

    rows = [r for r in rows
            if r.get("selection") in (args.selection, None)
            and (args.arms is None or r["arm"] in args.arms)]
    if not rows:
        raise SystemExit(f"no rows matching selection={args.selection} in {scores}")

    out_dir = Path(args.output_dir or scores.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        for key, (label, role, stat) in METRICS.items():
            series = defaultdict(list)
            for r in arm_rows:
                series[r["model"]].append((r["step"], r[role][stat]))
            if not series:
                continue
            title = f"{label} ({arm})"
            for model, points in sorted(series.items()):
                draw({model: points}, f"{title} - {model}", label,
                     out_dir / f"{arm}_{key}_{model}.png")
            if len(series) > 1:
                draw(dict(series), title, label, out_dir / f"{arm}_{key}.png")


if __name__ == "__main__":
    main()

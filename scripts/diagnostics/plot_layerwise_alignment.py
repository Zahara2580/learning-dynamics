"""
Plot layer-wise alignment: x = encoder layer, one line per checkpoint,
one figure per (model, metric). Reads results/alignment_layers/*.jsonl.

Usage:
    uv run python3 -m scripts.diagnostics.plot_layerwise_alignment
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = {
    "paired_cos": "Paired cosine",
    "random_cos": "Random cosine",
    "margin": "Margin",
    "retrieval_e2x": "Retrieval accuracy Eng to Xho",
    "retrieval_x2e": "Retrieval accuracy Xho to Eng",
}


def main() -> None:
    directory = Path("results/alignment_layers")
    rows = []
    for f in sorted(directory.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows under {directory}")

    out = directory / "plots"
    out.mkdir(exist_ok=True)

    models = sorted({r["model"] for r in rows})
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        steps = sorted({r["step"] for r in model_rows})
        colours = plt.cm.viridis([i / max(1, len(steps) - 1) for i in range(len(steps))])
        for key, label in METRICS.items():
            series = defaultdict(list)
            for r in model_rows:
                if key in r:
                    series[r["step"]].append((r["layer"], r[key]))
            if not series:
                continue
            figure, axis = plt.subplots(figsize=(8, 5))
            for step, colour in zip(steps, colours):
                points = sorted(series.get(step, []))
                if points:
                    axis.plot([l for l, _ in points], [v for _, v in points],
                              marker="o", markersize=3,
                              color="black" if step == 0 else colour,
                              linestyle="--" if step == 0 else "-",
                              label="base" if step == 0 else f"step {step}")
            axis.set_xlabel("Encoder layer")
            axis.set_ylabel(label)
            axis.set_title(f"{label} by layer for {model}")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
            figure.tight_layout()
            path = out / f"{model}_{key}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()

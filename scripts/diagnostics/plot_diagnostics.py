"""
Plot the diagnostic sweeps: cross-lingual alignment and zero-shot metrics
per CPT checkpoint.

Auto-detects whatever exists: results/alignment/*.jsonl and
results/zero_shot/*_*.jsonl. Step 0 (un-adapted model) is drawn as a
dashed reference line, linear CPT-step axis - same conventions as the
finetuning plots.

Usage:
    uv run python3 -m scripts.diagnostics.plot_diagnostics
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]

ALIGNMENT_METRICS = {
    "margin": "Cross-lingual margin",
    "retrieval_e2x": "Retrieval accuracy Eng to Xho",
    "retrieval_x2e": "Retrieval accuracy Xho to Eng",
    "paired_cos": "Paired cosine",
}
ZEROSHOT_METRICS = {"chrf": "chrF", "bleu": "BLEU", "zero_shot_loss": "Loss"}


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
            axis.axhline(base[0], color=c, linestyle="--", linewidth=1.0, alpha=0.6)
            if cpt:
                axis.text(cpt[-1][0], base[0], f" {model} base", color=c,
                          va="bottom", ha="right", fontsize=7)
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


def load_rows(directory: Path) -> list[dict]:
    rows = []
    for f in sorted(directory.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_alignment(directory: Path) -> None:
    rows = load_rows(directory)
    if not rows:
        return
    out = directory / "plots"
    out.mkdir(exist_ok=True)
    for key, label in ALIGNMENT_METRICS.items():
        series = defaultdict(list)
        for r in rows:
            if key in r:
                series[r["model"]].append((r["step"], r[key]))
        if series:
            draw(dict(series), label, label, out / f"{key}.png")


def plot_zero_shot(directory: Path) -> None:
    rows = load_rows(directory)
    if not rows:
        return
    out = directory / "plots"
    out.mkdir(exist_ok=True)
    tasks = {r.get("task", "d2t") for r in rows}
    for task in sorted(tasks):
        for key, label in ZEROSHOT_METRICS.items():
            series = defaultdict(list)
            for r in rows:
                if r.get("task", "d2t") != task:
                    continue
                value = r["metrics"].get(key) if key in ("chrf", "bleu") else r.get(key)
                if value is not None:
                    series[r["model"]].append((r["step"], value))
            if series:
                draw(dict(series), f"Zero-shot {label} ({task.upper()})", label,
                     out / f"{task}_{key}.png")


def main() -> None:
    plot_alignment(Path("results/alignment"))
    plot_alignment(Path("results/alignment_words"))
    plot_zero_shot(Path("results/zero_shot"))


if __name__ == "__main__":
    main()

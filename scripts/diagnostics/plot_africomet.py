"""
AfriCOMET score vs CPT step, one figure per arm (finetuned 3-epoch,
5-epoch en->xh, 5-epoch xh->en, zero-shot per direction).

Reads results/africomet/scores.jsonl. Arms with two selections are
filtered to last_epoch; the 3-epoch arm predates selections and plots
as-is. Step 0 is the dashed base reference - same conventions as the
other diagnostic plots.

Usage:
    uv run python3 -m scripts.diagnostics.plot_africomet
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]
ARM_TITLES = {
    ("ft-3ep", "mt"): "AfriCOMET, finetuned MT 3-epoch (en-xh)",
    ("ft-5ep", "mt"): "AfriCOMET, finetuned MT 5-epoch (en-xh)",
    ("ft-5ep-xhen", "mt-xhen"): "AfriCOMET, finetuned MT 5-epoch (xh-en)",
    ("zeroshot", "mt"): "AfriCOMET, zero-shot MT (en-xh)",
    ("zeroshot", "mt-xhen"): "AfriCOMET, zero-shot MT (xh-en)",
}


def colour(model: str) -> str:
    return f"C{MODEL_ORDER.index(model)}" if model in MODEL_ORDER else "C7"


def draw(series: dict[str, list[tuple[int, float]]], title: str, path: Path) -> None:
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
    axis.set_ylabel("AfriCOMET")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    if len(series) > 1:
        axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    scores = Path("results/africomet/scores.jsonl")
    rows = [json.loads(l) for l in scores.read_text().splitlines() if l.strip()]
    out = scores.parent / "plots"
    out.mkdir(exist_ok=True)

    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("selection") == "best_epoch":
            continue                      # last_epoch only where selections exist
        groups[(r["arm"], r["task"])][r["model"]].append((r["step"], r["africomet_mean"]))

    for key, series in sorted(groups.items()):
        title = ARM_TITLES.get(key, f"AfriCOMET {key[0]} {key[1]}")
        # per model, own y-scale - the absolute levels differ so much
        # (t5 far below the byte models) that a shared axis flattens
        # every trend; read these for patterns
        for model, points in sorted(series.items()):
            draw({model: points}, f"{title} - {model}",
                 out / f"{key[0]}_{key[1]}_{model}.png")
        # all models together - cross-model comparison only
        if len(series) > 1:
            draw(dict(series), title, out / f"{key[0]}_{key[1]}.png")


if __name__ == "__main__":
    main()

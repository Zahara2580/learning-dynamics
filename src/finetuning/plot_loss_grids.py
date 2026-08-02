"""
Per-checkpoint finetuning loss grids: one image per (model, task), with a
small train/eval-loss panel for each checkpoint.

Shows where each run's validation loss bottomed out (the selected epoch,
marked) and how far train and eval diverged - the view for spotting
overfitting and unstable runs across the sweep.

train_loss_per_epoch was added to the results row later than
val_loss_per_epoch, so older rows plot eval only.

Usage:
    uv run python3 -m src.finetuning.plot_loss_grids
    uv run python3 -m src.finetuning.plot_loss_grids --results results/variance/results.jsonl \
        --output-dir results/variance/plots
"""

import argparse
import json
import math
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Grid of per-checkpoint finetuning loss curves.")
    parser.add_argument("--results", type=str, default="results/finetune/results.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/finetune/plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--share-y", action="store_true",
                        help="Put every panel on one y-scale (easier to compare, "
                             "harder to see within-run shape).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["seed"] == args.seed]
    if not rows:
        raise SystemExit(f"no rows with seed {args.seed} in {args.results}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["task"])].append(row)

    for (model, task), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["ckpt_step"])
        ncols = args.ncols
        nrows = math.ceil(len(rs) / ncols)
        figure, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows),
                                    squeeze=False, sharey=args.share_y)
        n_with_train = 0

        for i, row in enumerate(rs):
            axis = axes[i // ncols][i % ncols]
            val = row.get("val_loss_per_epoch") or []
            train = row.get("train_loss_per_epoch") or []
            epochs = range(1, len(val) + 1)

            if train:
                n_with_train += 1
                axis.plot(range(1, len(train) + 1), train, color="C0",
                          marker="o", markersize=3, linewidth=1.2, label="train")
            if val:
                axis.plot(epochs, val, color="C3", marker="s", markersize=3,
                          linewidth=1.2, label="eval")
                best = row.get("best_epoch")
                if best:
                    axis.axvline(best, color="grey", linestyle=":", linewidth=1)

            step = row["ckpt_step"]
            axis.set_title("base" if step == 0 else f"step {step}", fontsize=9)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.25)

        # blank any unused cells
        for j in range(len(rs), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="upper right", fontsize=9)
        note = "" if n_with_train else "   (eval only - rows predate train-loss capture)"
        figure.suptitle(f"Finetuning loss per checkpoint - {model} / {task}{note}", fontsize=12)
        figure.supxlabel("epoch", fontsize=10)
        figure.supylabel("loss", fontsize=10)
        figure.tight_layout(rect=(0.01, 0.01, 1, 0.97))

        path = output_dir / f"loss_grid_{task}_{model}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"wrote {path}  ({len(rs)} checkpoints, {n_with_train} with train loss)")


if __name__ == "__main__":
    main()

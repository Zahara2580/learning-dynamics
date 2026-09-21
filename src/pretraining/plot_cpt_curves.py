"""Plot training and validation losses from saved CPT metrics."""

import argparse
import json
from argparse import Namespace
from pathlib import Path

from src.pretraining.schedule import CheckpointScheduleConfig, compute_checkpoint_steps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {
    "t5": "t5/metrics_t5_lafand_bs8.jsonl",
    "byt5": "byt5/metrics_byt5_lafand_bs4.jsonl",
    "nguni-byt5": "nguni-byt5/metrics_nguni-byt5_lafand_bs4.jsonl",

    "t5-bilingual": "t5/metrics_t5_lafand_bilingual.jsonl",
    "byt5-bilingual": "byt5/metrics_byt5_lafand_bilingual.jsonl",
    "nguni-byt5-bilingual": "nguni-byt5/metrics_nguni-byt5_lafand_bilingual.jsonl",
}
from src.finetuning.figstyle import (
    bold_axes, colour as _colour, display_name, fs, size,
)

COLOURS = {m: _colour(m) for m in MODELS}


def parse_args(argv=None) -> Namespace:
    parser = argparse.ArgumentParser(description="Plot CPT train/eval loss curves.")
    parser.add_argument('--results-root', type=str, default='/scratch/rmdrak003/results')
    parser.add_argument('--output-dir', type=str, default='results/cpt_plots')
    parser.add_argument(
        '--smooth',
        type=int,
        default=1,
        help='Optional smoothing window; defaults to unchanged logged values.',
    )
    parser.add_argument('--max-step', type=int, default=None, help='Truncate the x-axis.')
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Models to plot; defaults to all available models.',
    )
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Do not mark the saved checkpoint steps.',
    )
    return parser.parse_args(argv)


def load_metrics(path: Path) -> tuple[dict[int, float], dict[int, float], dict]:
    """Load training and evaluation losses, keeping the last record for each step."""
    train, evals = {}, {}

    n_lines, dup_train, dup_eval = 0, 0, 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step")
        if step is None:
            continue
        n_lines += 1
        if "loss" in row:
            dup_train += step in train
            train[step] = row["loss"]
        for key, value in row.items():
            if key.startswith("eval_") and key.endswith("_loss") \
                    and isinstance(value, (int, float)):
                series_ = evals.setdefault(key, {})
                dup_eval += step in series_
                series_[step] = value
    stats = {"lines": n_lines, "train_pts": len(train),
             "eval_series": {k: len(v) for k, v in evals.items()},
             "train_overlap": dup_train, "eval_overlap": dup_eval}
    return train, evals, stats


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    return [sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))]


def series(points: dict[int, float], max_step: int | None) -> tuple[list[int], list[float]]:
    steps = sorted(s for s in points if max_step is None or s <= max_step)
    return steps, [points[s] for s in steps]


def draw(curves: dict[str, dict[int, float]], title: str, path: Path, smooth: int, max_step: int | None, marker: bool, checkpoint_steps: list[int] | None = None, x_start: int | None = None) -> None:
    """Plot one loss curve per model and save the figure."""
    figure, axis = plt.subplots(figsize=size(9.5, 5.4))
    first, last = None, None
    for model, points in curves.items():
        steps, values = series(points, max_step)
        if not steps:
            continue
        smoothed = rolling_mean(values, smooth)

        ls = "--" if model.endswith("[eng]") else "-"
        axis.plot(steps, smoothed, color=COLOURS.get(model, "#555555"),
                  linestyle=ls, linewidth=2.0, label=display_name(model), zorder=3)
        first = steps[0] if first is None else min(first, steps[0])
        last = steps[-1] if last is None else max(last, steps[-1])

    if first is not None:
        left = first if x_start is None else x_start
        axis.set_xlim(left, last)
        ticks = [t for t in axis.get_xticks() if left < t <= last]
        axis.set_xticks([left] + ticks)
        axis.set_xlim(left, last)
    axis.set_xlabel("CPT step", fontsize=fs("label"), fontweight="bold")
    axis.set_ylabel("Loss", fontsize=fs("label"), fontweight="bold")
    axis.set_title(title, fontsize=fs("title"), fontweight="bold")
    axis.grid(alpha=0.3)
    bold_axes(axis)
    handles, labels = axis.get_legend_handles_labels()
    if handles:

        leg = axis.legend(handles, labels, loc="upper right",
                          fontsize=fs("legend"), frameon=True, fancybox=False,
                          edgecolor="0.35", framealpha=1.0, borderpad=0.7)
        leg.get_frame().set_linewidth(1.4)
        for t in leg.get_texts():
            t.set_fontweight("bold")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    print(f"wrote {path}")


def main(argv=None) -> None:
    """Create per-model and combined CPT loss figures."""
    args = parse_args(argv)
    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_losses, evals = {}, {}
    wanted = set(args.models) if args.models else None
    for model, rel in MODELS.items():
        if wanted is not None and model not in wanted:
            continue
        path = root / rel
        if not path.exists():
            print(f"Missing metrics: {path}")
            continue
        train, ev, st = load_metrics(path)
        training_losses[model] = train

        for key, points in ev.items():
            tag = key[len("eval_"):-len("_loss")]
            name = model if tag == "" else f"{model} [{tag}]"
            evals[name] = points
            COLOURS.setdefault(name, COLOURS[model])
        last = max(max(train, default=0),
                   max((max(v, default=0) for v in ev.values()), default=0))
        print(f"Model: {model}, last step: {last}")
        print(f"Records: {st['lines']}, training points: {st['train_pts']}")
        print(f"Evaluation points: {st['eval_series']}")
        print(f"Duplicate training records: {st['train_overlap']}")
        print(f"Duplicate evaluation records: {st['eval_overlap']}")

    if not training_losses:
        raise SystemExit(f"no metrics files found under {root}")

    ckpts = None if args.no_checkpoints else compute_checkpoint_steps(
        10_000, CheckpointScheduleConfig())

    panels = [
        ("train_loss", training_losses, "Training loss", args.smooth, False, 0),
        ("eval_loss", evals, "Validation loss", 1, True, None),
    ]

    for prefix, data, title, smooth, marker, x_start in panels:

        for model, points in data.items():
            safe = model.replace(" [", "_").replace("]", "")
            draw({model: points}, f"{title} for {model}",
                 output_dir / f"{prefix}_{safe}.png", smooth, args.max_step, marker,
                 ckpts, x_start)

        draw(data, title, output_dir / f"{prefix}_all.png", smooth, args.max_step, marker,
             ckpts, x_start)


if __name__ == "__main__":
    main()

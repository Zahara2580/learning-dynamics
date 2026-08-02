"""
CPT learning curves from the metrics_*.jsonl files.

Writes, per model and combined:
    train_loss_normalised_*   train loss / accumulation steps (comparable across models)
    train_loss_raw_*          train loss exactly as HF logged it (summed over the window)
    eval_loss_*               validation loss (already a correct mean - never normalised)

Chained jobs resume from the last checkpoint, so steps between that
checkpoint and the crash are logged twice. Dedupe keeps the LAST entry
per step: the file is append-ordered, so the final occurrence belongs to
the run that continued to 10k.

Usage:
    uv run python3 -m src.pretraining.plot_cpt_curves
    uv run python3 -m src.pretraining.plot_cpt_curves --smooth 1 --max-step 10000
"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

from src.pretraining.schedule import CheckpointScheduleConfig, compute_checkpoint_steps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# model -> (metrics filename, gradient accumulation steps of the winning arm)
MODELS = {
    "t5": ("t5/metrics_t5_lafand_bs8.jsonl", 128),
    "byt5": ("byt5/metrics_byt5_lafand_bs4.jsonl", 256),
    "nguni-byt5": ("nguni-byt5/metrics_nguni-byt5_lafand_bs4.jsonl", 256),
}
COLOURS = {m: f"C{i}" for i, m in enumerate(MODELS)}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Plot CPT train/eval loss curves.")
    parser.add_argument("--results-root", type=str, default="/scratch/rmdrak003/results")
    parser.add_argument("--output-dir", type=str, default="results/cpt_plots")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Rolling-mean window for train loss (1 = raw).")
    parser.add_argument("--max-step", type=int, default=None, help="Truncate the x-axis.")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Do not mark the saved checkpoint steps.")
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[dict[int, float], dict[int, float], dict]:
    """Read one metrics file into {step: loss} for train and eval; later entries win."""
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
        if "eval_loss" in row:
            dup_eval += step in evals
            evals[step] = row["eval_loss"]
    stats = {"lines": n_lines, "train_pts": len(train), "eval_pts": len(evals),
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


def draw(curves: dict[str, dict[int, float]], title: str, path: Path,
         smooth: int, max_step: int | None, marker: bool,
         checkpoint_steps: list[int] | None = None,
         x_start: int | None = None) -> None:
    """One figure; one line per model in curves. Titles stay short - the
    filename records whether it is the normalised or raw variant.

    Checkpoints are shown in two layers so nothing is invented: a grey
    vertical line marks every saved checkpoint (position only), and a
    diamond sits on the curve wherever a value was actually logged at that
    step. Train logs every step so all 19 get diamonds; eval logs every 200
    so five (100/300/500/700/900) have a line but no diamond.
    """
    figure, axis = plt.subplots(figsize=(9, 5))
    first, last = None, None
    for model, points in curves.items():
        steps, values = series(points, max_step)
        if not steps:
            continue
        smoothed = rolling_mean(values, smooth)
        axis.plot(steps, smoothed, color=COLOURS[model], marker="o" if marker else None,
                  markersize=3, linewidth=1.4, label=model, zorder=3)
        if checkpoint_steps:
            at = dict(zip(steps, smoothed))
            hits = [s for s in checkpoint_steps if s in at]
            axis.plot(hits, [at[s] for s in hits], linestyle="none", marker="D",
                      markersize=5, color=COLOURS[model], markeredgecolor="black",
                      markeredgewidth=0.6, zorder=4)
        first = steps[0] if first is None else min(first, steps[0])
        last = steps[-1] if last is None else max(last, steps[-1])

    # Position-only markers: no y-value is claimed, so the five checkpoints
    # without an eval measurement are still visible.
    if checkpoint_steps and first is not None:
        for i, step in enumerate([s for s in checkpoint_steps if first <= s <= last]):
            axis.axvline(step, color="grey", alpha=0.25, linewidth=0.8, zorder=0,
                         label="checkpoint" if i == 0 else None)

    # Span exactly the logged range, and force a tick on the left edge: the
    # default ticks omit it, which makes eval (which begins at step 200) look
    # like it begins at 0.
    if first is not None:
        left = first if x_start is None else x_start
        axis.set_xlim(left, last)
        ticks = [t for t in axis.get_xticks() if left < t <= last]
        axis.set_xticks([left] + ticks)
        axis.set_xlim(left, last)
    axis.set_xlabel("CPT step")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    # keep the checkpoint key even on single-model plots, where the title
    # already names the model
    if len(curves) == 1:
        keep = [(h, l) for h, l in zip(handles, labels) if l == "checkpoint"]
        handles, labels = ([h for h, _ in keep], [l for _, l in keep])
    if handles:
        axis.legend(handles, labels, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_train, norm_train, evals = {}, {}, {}
    print(f"{'model':>12} {'lines':>7} {'train':>7} {'eval':>6} {'tr_dup':>7} {'ev_dup':>7} {'last':>7}")
    for model, (rel, accum) in MODELS.items():
        path = root / rel
        if not path.exists():
            print(f"{model:>12}   MISSING {path}")
            continue
        train, ev, st = load_metrics(path)
        raw_train[model] = train
        norm_train[model] = {s: v / accum for s, v in train.items()}
        evals[model] = ev
        last = max(max(train, default=0), max(ev, default=0))
        print(f"{model:>12} {st['lines']:>7} {st['train_pts']:>7} {st['eval_pts']:>6} "
              f"{st['train_overlap']:>7} {st['eval_overlap']:>7} {last:>7}")

    if not raw_train:
        raise SystemExit(f"no metrics files found under {root}")

    ckpts = None if args.no_checkpoints else compute_checkpoint_steps(
        10_000, CheckpointScheduleConfig())

    panels = [
        ("train_loss_normalised", norm_train, "Training loss", args.smooth, False, 0),
        ("train_loss_raw", raw_train, "Training loss", args.smooth, False, 0),
        ("eval_loss", evals, "Validation loss", 1, True, None),
    ]

    for prefix, data, title, smooth, marker, x_start in panels:
        # one figure per model
        for model, points in data.items():
            draw({model: points}, f"{title} for {model}",
                 output_dir / f"{prefix}_{model}.png", smooth, args.max_step, marker,
                 ckpts, x_start)
        # all models together
        draw(data, title, output_dir / f"{prefix}_all.png", smooth, args.max_step, marker,
             ckpts, x_start)


if __name__ == "__main__":
    main()

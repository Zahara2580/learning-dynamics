"""Draw fine-tuning and alignment figures and dispatch other plot commands."""

import argparse
import csv
import json
import importlib
import string
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from src.finetuning.figstyle import (
    BASE_SWATCH, MODEL_ORDER, base_model, bold_axes, boxed_legend, colour,
    display_name, fs, model_key, size,
)

SPECIALIZED = {
    "probes": ("src.diagnostics.plot_new_probes", "POS accuracy, pooling and per-class F1"),
    "cpt": ("src.pretraining.plot_cpt_curves", "CPT training/validation losses"),
    "f1": ("src.diagnostics.summarize_t2x_f1", "T2X subject/object F1 tables and figures"),
}


LABELS = {"bleu": "BLEU", "chrf": "chrF", "chrf_pp": "chrF++", "ter": "TER"}
SPARSE_STEPS = [0, 100] + list(range(1000, 10001, 1000))
EARLY_STEPS = [0] + list(range(100, 1001, 100))


def load_scores(path, seed, selection, steps=None):
    """Load scores for the selected seed, epoch selection and checkpoint steps."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]
    rows = [r for r in rows if r["seed"] == seed
            and r.get("selection", selection) == selection]
    if steps is not None:
        keep = set(steps) | {0}
        rows = [r for r in rows if r["ckpt_step"] in keep]
    return rows


def groups_for(curves, forced, never):
    """Split curves across the widest qualifying gap between their value ranges."""
    spans = {m: (min(v for _, v in p), max(v for _, v in p))
             for m, p in curves.items()}
    order = sorted(spans, key=lambda m: spans[m][1], reverse=True)
    if never or len(order) < 2:
        return order, []

    if forced is not None:
        lower = [m for m in order if spans[m][1] < forced]
        upper = [m for m in order if spans[m][1] >= forced]
        return (upper, lower) if lower and upper else (order, [])

    best = None
    for i in range(1, len(order)):
        upper, lower = order[:i], order[i:]
        band = min(spans[m][0] for m in upper) - max(spans[m][1] for m in lower)
        width = lambda g: max(spans[m][1] for m in g) - min(spans[m][0] for m in g)
        narrower = min(width(upper), width(lower))
        if band > 0 and band > 1.5 * narrower and (best is None or band > best[0]):
            best = (band, upper, lower)
    return (best[1], best[2]) if best else (order, [])


def smooth_path(xs, ys, mode):
    """Return connecting-line coordinates, optionally using shape-preserving interpolation."""
    if mode != "pchip" or len(xs) < 3:
        return xs, ys
    from scipy.interpolate import PchipInterpolator
    import numpy as np
    f = PchipInterpolator(xs, ys)
    dense = np.linspace(min(xs), max(xs), 400)
    return dense, f(dense)


@dataclass
class Panel:
    """Describe a grid panel’s data, metric, title and position."""
    rows: list
    metric: str
    title: str
    row: int
    col: int
    letter: str


def build_parser():
    """Build the command-line parser for all figure families."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="figure", required=True)
    for name, description in [("full", "Full-data MT/D2T 2x3 figure"),
                              ("ablations", "Data-size ablations, 2x3 per metric"),
                              ("task", "Custom task panels in column order")]:
        p = sub.add_parser(name, help=description, description=description)
        if name in {"full", "ablations"}:
            for flag in ("--mt", "--mt-xhen", "--d2t"):
                p.add_argument(flag, required=True, **({"nargs": 2} if name == "ablations" else {}))
            if name == "full":
                p.add_argument('--metric', nargs=2, default=['chrf', 'bleu'], choices=list(LABELS))
            else:
                p.add_argument('--metric', required=True, choices=list(LABELS))
                p.add_argument('--sizes', nargs=2, default=['10k pairs', '25k pairs'])
                p.add_argument('--d2t-sizes', nargs=2, default=['1,000 examples', '2,000 examples'])
        else:
            p.add_argument(
                '--panel',
                action='append',
                required=True,
                help='file.jsonl:metric:Title; column-major order',
            )
            p.add_argument('--rows', type=int, required=True)
            p.add_argument('--divider-after', type=int)
            p.add_argument('--title')
        p.add_argument('--out', required=True)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--selection', default='last_epoch')
        p.add_argument('--pad', type=float, default=0.12)
        sampling = p.add_mutually_exclusive_group()
        sampling.add_argument('--steps', type=int, nargs='+')
        sampling.add_argument('--sparse', action='store_true')
        sampling.add_argument('--early', action='store_true')
        p.add_argument('--no-break', action='store_true')
    for name, (_, description) in {**LOCAL_PLOTS, **SPECIALIZED}.items():
        sub.add_parser(name, help=description, add_help=False)
    return parser


def build_panels(a):
    """Load the requested results and arrange them into figure panels."""
    steps = EARLY_STEPS if a.early else SPARSE_STEPS if a.sparse else a.steps

    def read(path):
        """Load matching score records and reject an empty selection."""
        rows = load_scores(path, a.seed, a.selection, steps)
        if not rows:
            raise ValueError(f"No rows for seed={a.seed}, selection={a.selection}: {path}")
        return rows

    if a.figure == "task":
        if a.rows < 1:
            raise ValueError("--rows must be positive")
        cells = []
        for i, spec in enumerate(a.panel):
            path, metric, title = spec.split(":", 2)
            if metric not in LABELS:
                raise ValueError(f"Unknown metric: {metric}")
            if i >= len(string.ascii_lowercase):
                raise ValueError("At most 26 lettered panels are supported")
            cells.append(Panel(read(path), metric, title, i % a.rows, i // a.rows, string.ascii_lowercase[i]))
        cols = -(-len(cells) // a.rows)
        if a.divider_after is not None and not 0 <= a.divider_after < cols - 1:
            raise ValueError("--divider-after must identify a column before the last column")
        return cells, a.rows, cols

    titles = {"mt": "MT en to xh", "mt_xhen": "MT xh to en", "d2t": "D2T"}
    files = {"mt": a.mt, "mt_xhen": a.mt_xhen, "d2t": a.d2t}
    if a.figure == "full":
        rows = {key: read(path) for key, path in files.items()}
        layout = [("mt", 0, 0, 0, "a"), ("mt", 1, 0, 1, "b"),
                  ("mt_xhen", 0, 1, 0, "c"), ("mt_xhen", 1, 1, 1, "d"),
                  ("d2t", 0, 0, 2, "e"), ("d2t", 1, 1, 2, "f")]
        cells = [Panel(rows[key], a.metric[k], f"{titles[key]} - {LABELS[a.metric[k]]}", r, c, letter)
                 for key, k, r, c, letter in layout]
    else:
        rows = {(key, i): read(path) for key, paths in files.items() for i, path in enumerate(paths)}
        layout = [("mt", 0, 0, 0, "a"), ("mt", 1, 0, 1, "c"),
                  ("mt_xhen", 0, 1, 0, "b"), ("mt_xhen", 1, 1, 1, "d"),
                  ("d2t", 0, 0, 2, "e"), ("d2t", 1, 1, 2, "f")]
        cells = [Panel(rows[key, k], a.metric,
                       f"{titles[key]}, {(a.d2t_sizes if key == 'd2t' else a.sizes)[k]}", r, c, letter)
                 for key, k, r, c, letter in layout]
    return cells, 2, 3


def curves_for(rows, metric):
    """Group metric values by model and sort them by checkpoint step."""
    curves = defaultdict(list)
    for row in rows:
        if metric in row["metrics"]:
            curves[row["model"]].append((row["ckpt_step"], row["metrics"][metric]))
    if not curves:
        raise ValueError(f"No {metric} scores in panel")
    return {model: sorted(points) for model, points in curves.items()}


def draw_group(ax, group, curves, a, xmax):
    """Draw model curves and baseline scores on one axis."""
    values = []
    for model in sorted(group, key=model_key):
        points = curves[model]
        base = [v for step, v in points if step == 0]
        cpt = [(step, v) for step, v in points if step > 0]
        color = colour(model)
        if base:
            ax.axhline(base[0], color=color, ls="--", lw=1.3, alpha=0.65)
            values.append(base[0])
        if cpt:
            xs, ys = zip(*cpt)
            ax.plot(xs, ys, lw=2.0, color=color, label=model, zorder=3, solid_joinstyle="miter")
            ax.plot(xs, ys, linestyle="none", marker="o", ms=4.0 if a.figure == "task" else 4.5,
                    color=color, zorder=4)
            values += ys
    if values:
        lo, hi = min(values), max(values)
        padding = (hi - lo) * a.pad or 0.5
        ax.set_ylim(lo - padding, hi + padding)
    ax.set_xlim(0, xmax * 1.02)
    ax.grid(alpha=0.3)
    if a.figure == "task":
        bold_axes(ax)
    else:
        ax.tick_params(labelsize=13, width=1.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")


def draw_panel(fig, grid, cell, a, xmax):
    """Draw a labelled metric panel with an optional axis break."""
    curves = curves_for(cell.rows, cell.metric)
    upper, lower = groups_for(curves, None, a.no_break)
    label_size = 15 if a.figure == "full" else fs("label")
    if lower:
        sub = grid.subgridspec(2, 1, height_ratios=[2, 1], hspace=0.10)
        hi, lo = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        draw_group(hi, upper, curves, a, xmax)
        draw_group(lo, lower, curves, a, xmax)
        hi.spines["bottom"].set_visible(False)
        lo.spines["top"].set_visible(False)
        hi.tick_params(bottom=False, labelbottom=False)
        marks = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
                     color="k", mec="k", mew=1.4, clip_on=False)
        hi.plot([0, 1], [0, 0], transform=hi.transAxes, **marks)
        lo.plot([0, 1], [1, 1], transform=lo.transAxes, **marks)
        lo.set_xlabel("CPT step", fontsize=label_size, fontweight="bold")
        hi.set_ylabel(LABELS[cell.metric], fontsize=label_size, fontweight="bold", y=0.25)
        top = hi
    else:
        top = fig.add_subplot(grid)
        draw_group(top, upper, curves, a, xmax)
        top.set_xlabel("CPT step", fontsize=label_size, fontweight="bold")
        top.set_ylabel(LABELS[cell.metric], fontsize=label_size, fontweight="bold")
    top.set_title(cell.title, fontsize=15 if a.figure == "full" else fs("title"), fontweight="bold", pad=8)
    x, y = (-0.17, 1.15) if a.figure == "task" else (-0.11, 1.12)
    top.text(x, y, f"({cell.letter})", transform=top.transAxes,
             fontsize=17 if a.figure == "full" else fs("letter"), fontweight="bold", va="top", ha="left")


def render_grid(a):
    """Render and save the selected fine-tuning grid layout."""
    cells, rows, cols = build_panels(a)
    xmax = max(row["ckpt_step"] for cell in cells for row in cell.rows)
    task = a.figure == "task"
    fig = plt.figure(figsize=size(4.9 * cols, 4.7 * rows + 1.2) if task else size(18.0, 9.4))
    if a.figure == "ablations":
        fig.suptitle(f"Data-size ablations: {LABELS[a.metric]}", fontsize=fs("suptitle"),
                     fontweight="bold", y=1.012)
    outer = fig.add_gridspec(rows, cols, hspace=0.46 if task else 0.42, wspace=0.34,
                             top=0.885 if task else 0.93, bottom=0.135 if task else 0.115,
                             left=0.048, right=0.988)
    for cell in cells:
        draw_panel(fig, outer[cell.row, cell.col], cell, a, xmax)
    if not task or a.divider_after is not None:
        fig.canvas.draw()
        if task:
            left = outer[0, a.divider_after].get_position(fig).x1
            right = outer[0, a.divider_after + 1].get_position(fig).x0
            x = left + 0.28 * (right - left)
        else:
            left = max(ax.get_position().x1 for ax in fig.axes if ax.get_position().x1 < 0.70)
            right = min(ax.get_position().x0 for ax in fig.axes if ax.get_position().x0 > 0.60)
            x = left + 0.25 * (right - left)
        y0 = min(ax.get_position().y0 for ax in fig.axes)
        y1 = max(ax.get_position().y1 for ax in fig.axes)
        fig.add_artist(Line2D([x, x], [y0, y1], transform=fig.transFigure, color="0.45", lw=2.6))
    handles = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    entries = [(handles[k], display_name(k)) for k in sorted(handles, key=model_key)]
    entries.append((Line2D([0], [0], color=BASE_SWATCH, ls="--", lw=1.8), "base model"))
    if task:
        boxed_legend(fig, [h for h, _ in entries], [label for _, label in entries], y=0.004)
        if a.title:
            fig.suptitle(a.title, fontsize=fs("suptitle"), fontweight="bold", y=0.975)
    else:
        legend = fig.legend([h for h, _ in entries], [label for _, label in entries],
                            ncol=len(entries), loc="lower center", fontsize=15 if a.figure == "full" else fs("label"),
                            frameon=True, fancybox=False, edgecolor="0.35", framealpha=1.0,
                            borderpad=0.7, columnspacing=2.4, bbox_to_anchor=(0.5, -0.055))
        legend.get_frame().set_linewidth(1.4)
        for label in legend.get_texts():
            label.set_fontweight("bold")
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {path}")


def alignment_colour(model: str) -> str:
    """Return the original alignment colour for a model family."""
    model = base_model(model)
    return f"C{MODEL_ORDER.index(model)}" if model in MODEL_ORDER else "C7"


ALIGNMENT_LABELS = {
    "cosine_mean": "Mean cosine",
    "baseline": "Mean cosine (all pairs)",
    "cosine_gap": "Cosine gap",
    "p_at_1_e2x": "P@1 (en to xh)",
    "p_at_1_x2e": "P@1 (xh to en)",
}
ALIGNMENT_SHORT = {
    "cosine_mean": "Mean cosine",
    "baseline": "All pairs",
    "cosine_gap": "Cosine gap",
    "p_at_1_e2x": "P@1 en to xh",
    "p_at_1_x2e": "P@1 xh to en",
}


ALIGNMENT_STEPS = [0, 1000, 3000, 4000, 5000, 6000, 10000]

STEP_COLOURS = {
    0: "#000000",
    1000: "#0072B2",
    3000: "#009E73",
    4000: "#E69F00",
    5000: "#D55E00",
    6000: "#CC79A7",
    10000: "#7B3294",
    2000: "#56B4E9",
    7000: "#8C8C00",
    8000: "#666666",
    9000: "#00688B",
}


def parse_alignment_args(argv=None) -> Namespace:
    """Parse options for alignment layer and checkpoint figures."""
    p = argparse.ArgumentParser(description="Cross-lingual alignment figures.")
    p.add_argument('mode', choices=['layers', 'steps', 'panels'])
    p.add_argument('--results-dir', type=str, default='alignment_results/all_alignment results')
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--models', type=str, nargs='+', default=MODEL_ORDER)
    p.add_argument(
        '--metric',
        type=str,
        nargs='+',
        default=['cosine_mean'],
        choices=list(ALIGNMENT_LABELS),
    )
    p.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=None,
        help='Default: the seven-checkpoint set in layers mode, every checkpoint in steps mode.',
    )
    p.add_argument('--layer-min', type=int, default=None, help='Lowest encoder layer to include.')
    p.add_argument(
        '--layer-frac',
        type=float,
        default=None,
        help='Fraction of lower encoder layers to exclude.',
    )
    p.add_argument(
        '--split',
        type=float,
        nargs='+',
        default=None,
        help='steps mode: force panel breaks at these values.',
    )
    p.add_argument('--no-break', action='store_true')
    p.add_argument('--max-groups', type=int, default=3)
    p.add_argument(
        '--per-model',
        action='store_true',
        help='Draw one axis panel per model in steps mode.',
    )
    p.add_argument('--pad', type=float, default=0.12)
    p.add_argument(
        '--smooth',
        choices=['round', 'pchip', 'none'],
        default='pchip',
        help='Connecting-line style; markers stay at measured values.',
    )
    p.add_argument(
        '--layout',
        choices=['grid', 'stack'],
        default='grid',
        help='Arrange panels as a model-by-metric grid or a single column.',
    )
    p.add_argument('--suffix', type=str, default='')
    return p.parse_args(argv)


def load_alignment(results_dir, model, layers):
    """Load saved alignment records for a model."""
    path = Path(results_dir) / (f"{model}_layers.jsonl" if layers else f"{model}.jsonl")
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def draw_alignment_layers(ax, rows, metric, steps, layer_min, layer_frac=None):
    """Draw checkpoint curves across encoder layers."""
    if layer_frac is not None:
        top = max(r["layer"] for r in rows)
        layer_min = max(layer_min or 0, round(layer_frac * top))
    per_step = defaultdict(list)
    for r in rows:
        if r["step"] in steps and metric in r:
            if layer_min is None or r["layer"] >= layer_min:
                per_step[r["step"]].append((r["layer"], r[metric]))
    for step in sorted(per_step):
        pts = sorted(per_step[step])
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        c = STEP_COLOURS.get(step, "#777777")
        ax.plot(xs, ys, lw=1.7, color=c, zorder=2,
                label="base" if step == 0 else f"{step:,}",
                ls="--" if step == 0 else "-",
                solid_joinstyle="round", solid_capstyle="round")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Encoder layer", fontsize=fs("label"), fontweight="bold")
    bold_axes(ax)
    return per_step


def alignment_layers(a):
    """Build a figure comparing alignment across layers and models."""
    metrics, models = a.metric, a.models
    nrows, ncols = len(metrics), len(models)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=size(4.8 * ncols, 3.3 * nrows + 1.3))
    handles = {}
    for j, model in enumerate(models):
        rows = load_alignment(a.results_dir, model, layers=True)
        for i, metric in enumerate(metrics):
            ax = axes[i][j]
            draw_alignment_layers(ax, rows, metric, set(a.steps or ALIGNMENT_STEPS), a.layer_min,
                        a.layer_frac)
            if i == 0:
                ax.set_title(display_name(model),
                             fontsize=fs("title"), fontweight="bold")
            if j == 0:
                ax.set_ylabel(ALIGNMENT_LABELS[metric], fontsize=fs("label"),
                              fontweight="bold")
            if i < nrows - 1:
                ax.set_xlabel("")
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)

    title = ("Cross-lingual alignment across encoder layers" if len(metrics) > 1
             else f"{ALIGNMENT_LABELS[metrics[0]]} across encoder layers")
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 1.12 / h, 1, 1 - 0.42 / h))
    fig.suptitle(title, fontsize=fs("suptitle"), fontweight="bold", y=1 - 0.10 / h)
    order = sorted(handles, key=lambda l: -1 if l == "base" else int(l.replace(",", "")))
    leg = boxed_legend(fig, [handles[l] for l in order], order, y=0.012)
    leg.set_title("CPT step", prop={"weight": "bold", "size": fs("legend")})
    return fig, f"alignment_layers_{'_'.join(metrics)}"


def alignment_groups(curves, forced, never, max_groups, per_model=False):
    """Group model curves into axis panels using their value ranges."""
    spans = {m: (min(v for _, v in p), max(v for _, v in p))
             for m, p in curves.items()}
    order = sorted(spans, key=lambda m: spans[m][1], reverse=True)
    if never or len(order) < 2:
        return [order]
    if per_model:
        return [[m] for m in order]
    if forced:
        cuts = sorted(forced, reverse=True)
        groups, rest = [], list(order)
        for cut in cuts:
            groups.append([m for m in rest if spans[m][1] >= cut])
            rest = [m for m in rest if spans[m][1] < cut]
        groups.append(rest)
        return [g for g in groups if g]

    groups = [[order[0]]]
    g_lo, g_hi = spans[order[0]]
    for m in order[1:]:
        lo, hi = spans[m]
        band = g_lo - hi
        if band > 0 and band > min(g_hi - g_lo, hi - lo) and len(groups) < max_groups:
            groups.append([m])
            g_lo, g_hi = lo, hi
        else:
            groups[-1].append(m)
            g_lo, g_hi = min(g_lo, lo), max(g_hi, hi)
    return groups


def draw_alignment_models(ax, group, curves, pad, smooth='round'):
    """Draw model alignment curves and base-model reference lines."""
    vals = []
    for m in group:
        pts = curves[m]
        base = [v for s, v in pts if s == 0]
        cpt = [(s, v) for s, v in pts if s > 0]
        c = alignment_colour(m)
        if base:
            ax.axhline(base[0], color=c, ls="--", lw=1.0, alpha=0.55)
            vals.append(base[0])
        if cpt:
            xs = [s for s, _ in cpt]
            ys = [v for _, v in cpt]
            px, py = smooth_path(xs, ys, smooth)
            ax.plot(px, py, lw=1.6, color=c, label=m, zorder=2,
                    solid_joinstyle="round", solid_capstyle="round")
            ax.plot(xs, ys, linestyle="none", marker="o", ms=3.5, color=c, zorder=3)
            vals += ys
    if vals:
        lo, hi = min(vals), max(vals)
        p = (hi - lo) * pad or abs(hi) * 0.05 or 0.01
        ax.set_ylim(lo - p, hi + p)
    ax.grid(alpha=0.3)
    bold_axes(ax)


def alignment_steps(a):
    """Build alignment curves over CPT steps with optional axis breaks."""
    per_metric = {}
    for metric in a.metric:
        c = defaultdict(list)
        for model in a.models:
            for r in load_alignment(a.results_dir, model, layers=False):
                if metric in r and (a.steps is None or r["step"] in set(a.steps)):
                    c[model].append((r["step"], r[metric]))
        per_metric[metric] = {m: sorted(p) for m, p in c.items() if p}

    splits = {}
    for metric, curves in per_metric.items():
        splits[metric] = alignment_groups(curves, a.split, a.no_break,
                                      a.max_groups, a.per_model)
    nrows = max(len(g) for g in splits.values())
    ncols = len(per_metric)

    fig = plt.figure(figsize=(5.2 * ncols, 1.9 * nrows + 1.55))
    fig_h = fig.get_figheight()
    gs = fig.add_gridspec(nrows, ncols, hspace=0.10, wspace=0.28,
                          top=1 - 0.72 / fig_h, bottom=0.82 / fig_h,
                          left=0.85 / (5.2 * ncols), right=0.99)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)

    handles = {}
    for col, (metric, curves) in enumerate(per_metric.items()):
        groups = splits[metric]
        spans, r = [], 0
        for k, g in enumerate(groups):
            h = nrows - r - (len(groups) - k - 1)
            h = h if k == len(groups) - 1 else 1
            spans.append((r, r + h))
            r += h
        axes = []
        for (r0, r1), g in zip(spans, groups):
            ax = fig.add_subplot(gs[r0:r1, col], sharex=axes[0] if axes else None)
            draw_alignment_models(ax, g, curves, a.pad, a.smooth)
            axes.append(ax)
        for k, ax in enumerate(axes):
            if k > 0:
                ax.spines["top"].set_visible(False)
                ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kw)
            if k < len(axes) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False, labelbottom=False)
                ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kw)
        axes[-1].set_xlabel("CPT step", fontsize=fs("label"), fontweight="bold")
        mid = axes[len(axes) // 2]
        mid.set_ylabel(ALIGNMENT_LABELS[metric], fontsize=fs("label"), fontweight="bold", y=0.5 if len(axes) % 2 else 1.0)
        if ncols > 1:
            axes[0].set_title(ALIGNMENT_SHORT[metric], fontsize=12)
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)

    labels = sorted(handles, key=lambda m: MODEL_ORDER.index(m)
                    if m in MODEL_ORDER else 99)
    boxed_legend(fig, [handles[l] for l in labels], labels, y=0.005)
    title = ("Cross-lingual alignment at the final encoder layer" if ncols > 1
             else f"{ALIGNMENT_LABELS[a.metric[0]]} at the final encoder layer")
    fig.suptitle(title, fontsize=fs("suptitle"), fontweight="bold", y=1 - 0.08 / fig_h)
    return fig, f"alignment_steps_{'_'.join(per_metric)}"


def alignment_panels(a):
    """Build separate alignment panels for each model and metric."""
    curves = {}
    for metric in a.metric:
        for model in a.models:
            pts = [(r["step"], r[metric]) for r in load_alignment(a.results_dir, model, layers=False)
                   if metric in r and (a.steps is None or r["step"] in set(a.steps))]
            if pts:
                curves[(metric, model)] = sorted(pts)

    stack = a.layout == "stack"
    cells = [(m, mo) for m in a.metric for mo in a.models if (m, mo) in curves]
    nrows, ncols = (len(cells), 1) if stack else (len(a.models), len(a.metric))
    w = 5.6 if stack else 4.8 * ncols
    fig = plt.figure(figsize=(w, (1.75 if stack else 2.35) * nrows + 1.15))
    fig_h = fig.get_figheight()
    gs = fig.add_gridspec(nrows, ncols,
                          hspace=0.42 if stack else 0.30, wspace=0.26,
                          top=1 - 0.72 / fig_h, bottom=0.45 / fig_h,
                          left=(0.95 if stack else 0.85) / w, right=0.99)

    for k, (metric, model) in enumerate(cells):
        r, c = (k, 0) if stack else (a.models.index(model), a.metric.index(metric))
        ax = fig.add_subplot(gs[r, c])
        draw_alignment_models(ax, [model], {model: curves[(metric, model)]},
                    a.pad, a.smooth)
        if stack:
            ax.set_title(display_name(model), fontsize=fs("title"), fontweight="bold")
            ax.set_ylabel(ALIGNMENT_LABELS[metric])
            if k == len(cells) - 1:
                ax.set_xlabel("CPT step", fontsize=fs("label"), fontweight="bold")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            if c == 0:
                ax.set_ylabel(display_name(model), fontsize=fs("label"), fontweight="bold")
            if r == 0:
                ax.set_title(ALIGNMENT_LABELS[metric], fontsize=fs("title"), fontweight="bold")
            if r == nrows - 1:
                ax.set_xlabel("CPT step", fontsize=fs("label"), fontweight="bold")

    fig.suptitle("Cross-lingual alignment at the final encoder layer",
                 fontsize=fs("suptitle"), fontweight="bold", y=1 - 0.08 / fig_h)
    return fig, f"alignment_panels_{'_'.join(a.metric)}_{a.layout}"


def plot_alignment(argv=None) -> None:
    """Render and save the requested alignment figure."""
    a = parse_alignment_args(argv)
    fig, name = {"layers": alignment_layers, "steps": alignment_steps,
                 "panels": alignment_panels}[a.mode](a)
    if a.mode == "layers" and a.layer_frac is not None:
        name += f"_top{round((1 - a.layer_frac) * 100)}pct"
    elif a.mode == "layers" and a.layer_min is not None:
        name += f"_from{a.layer_min}"
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}{a.suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"  wrote {path}")


BILINGUAL_MODELS = ["t5-bilingual", "byt5-bilingual", "nguni-byt5-bilingual"]
BILINGUAL_TITLES = {"t5-bilingual": "T5", "byt5-bilingual": "ByT5",
         "nguni-byt5-bilingual": "Nguni-ByT5"}
BILINGUAL_ROWS = [("probe", "Probing accuracy"),
        ("cosine_mean", "Mean cosine"),
        ("cosine_gap", "Cosine gap")]


def parse_bilingual_args(argv=None) -> Namespace:
    """Parse options for the combined bilingual figure."""
    p = argparse.ArgumentParser(description="Bilingual probing + alignment, 3x3.")
    p.add_argument('--alignment-dir', default='alignment_results/bilingual')
    p.add_argument('--scores', default='new_probes/bilingual/scores (1).csv')
    p.add_argument('--pooling', default='mean')
    p.add_argument('--out', required=True)
    return p.parse_args(argv)


def load_bilingual(a):
    """Load bilingual alignment records and POS probe scores."""
    al = defaultdict(dict)
    for f in Path(a.alignment_dir).glob("*_layers.jsonl"):
        for line in f.read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                al[(r["model"], r["step"])][r["layer"]] = r
    pr = defaultdict(dict)
    for r in csv.DictReader(open(a.scores)):
        if r["split"] == "test" and r["pooling"] == a.pooling:
            pr[(r["model"], int(r["step"]))][int(r["layer"])] = float(r["accuracy"])
    return al, pr


def plot_bilingual(argv=None) -> None:
    """Render the combined bilingual probing and alignment figure."""
    from src.diagnostics.plot_new_probes import SHADOW_STEP, STEP_MARKER

    a = parse_bilingual_args(argv)
    al, pr = load_bilingual(a)

    fig = plt.figure(figsize=size(16.5, 12.4))
    LEFT, RIGHT = 0.058, 0.988
    gs_top = fig.add_gridspec(1, 3, left=LEFT, right=RIGHT,
                              top=0.925, bottom=0.735, wspace=0.24)
    gs_bot = fig.add_gridspec(2, 3, left=LEFT, right=RIGHT,
                              top=0.600, bottom=0.125, wspace=0.24, hspace=0.30)

    blocks = [(gs_top, [BILINGUAL_ROWS[0]]), (gs_bot, BILINGUAL_ROWS[1:])]
    first_ax = None
    for gs, rows in blocks:
        for r, (kind, ylab) in enumerate(rows):
            for c, model in enumerate(BILINGUAL_MODELS):
                ax = fig.add_subplot(gs[r, c])
                first_ax = first_ax or ax
                for step in ALIGNMENT_STEPS:
                    src = (pr.get((model, step)) if kind == "probe"
                           else al.get((model, step)))
                    if not src:
                        continue
                    xs = sorted(src)
                    ys = [src[x] if kind == "probe" else src[x][kind] for x in xs]
                    colour = STEP_COLOURS.get(step, "#777")
                    ax.plot(xs, ys, lw=1.8, color=colour,
                            ls="--" if step == 0 else "-",
                            label="base" if step == 0 else f"{step:,}")
                    if kind != "probe":
                        continue
                    st = STEP_MARKER.get(step, dict(marker="*", ms=11, mew=0.6))
                    bi = max(range(len(xs)), key=lambda k: ys[k])
                    ax.plot(xs[bi], ys[bi], linestyle="none",
                            marker=st["marker"], ms=st["ms"] * 0.85,
                            markeredgewidth=st["mew"],
                            markerfacecolor="none" if step == 0 else colour,
                            markeredgecolor=colour if step == 0 else "black",
                            zorder=8 if step == 0 else 6,
                            path_effects=[pe.SimplePatchShadow(
                                offset=(2, -2), shadow_rgbFace="k", alpha=0.45),
                                pe.Normal()] if step == SHADOW_STEP else None)
                ax.grid(alpha=0.3)
                bold_axes(ax)
                if gs is gs_top or r == 0:
                    ax.set_title(BILINGUAL_TITLES[model], fontsize=fs("title"),
                                 fontweight="bold")
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=fs("label"), fontweight="bold")
                if r == len(rows) - 1:
                    ax.set_xlabel("Encoder layer", fontsize=fs("label"),
                                  fontweight="bold")

    fig.text(0.5, 0.962, "(a)  POS probing accuracy", ha="center", va="center",
             fontsize=fs("suptitle"), fontweight="bold")
    fig.add_artist(Line2D([LEFT, RIGHT], [0.665, 0.665], transform=fig.transFigure,
                          color="0.45", lw=2.6))
    fig.text(0.5, 0.634, "(b)  Cross-lingual alignment", ha="center", va="center",
             fontsize=fs("suptitle"), fontweight="bold")

    handles, labels = [], []
    for step in ALIGNMENT_STEPS:
        st = STEP_MARKER.get(step, dict(marker="*", ms=11, mew=0.6))
        c = STEP_COLOURS.get(step, "#777")
        handles.append(Line2D([0], [0], color=c, lw=1.8,
                              ls="--" if step == 0 else "-",
                              marker=st["marker"], ms=min(st["ms"], 11) * 0.85,
                              markerfacecolor="none" if step == 0 else c,
                              markeredgecolor=c if step == 0 else "black",
                              markeredgewidth=st["mew"]))
        labels.append("base" if step == 0 else f"{step:,}")
    leg = boxed_legend(fig, handles, labels, y=0.004)
    leg.set_title("CPT step", prop={"weight": "bold", "size": fs("legend")})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  wrote {out}")


LOCAL_PLOTS = {
    "alignment": (plot_alignment, "Alignment layer/step panels"),
    "bilingual": (plot_bilingual, "Combined bilingual probing/alignment figure"),
}


def main(argv=None):
    """Run the requested figure command."""
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.figure in LOCAL_PLOTS:
        render, _ = LOCAL_PLOTS[args.figure]
        return render(remaining)
    if args.figure in SPECIALIZED:
        module, _ = SPECIALIZED[args.figure]
        return importlib.import_module(module).main(remaining)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    try:
        render_grid(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

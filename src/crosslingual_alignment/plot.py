"""Plot checkpoint trajectories and layer profiles from saved alignment scores."""

# %pip install -q matplotlib

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_ORDER = ["t5", "byt5", "nguni-byt5"]


def read_jsonl(path):
    """Read saved alignment records."""
    return [json.loads(l) for l in path.read_text().split('\n') if l.strip()]


def plot_results(input_dir: Path) -> None:
    """Draw the trajectory and layer figures from alignment JSONL files."""
    OUT = input_dir
    PLOTS = OUT / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = [r for p in sorted(OUT.glob('*.jsonl')) if '_layers' not in p.name for r in read_jsonl(p)]
    for key, label in [
        ('cosine_gap', 'Cosine gap'),
        ('p_at_1_e2x', 'P@1 eng to xho'),
        ('p_at_1_x2e', 'P@1 xho to eng'),
        ('cosine_mean', 'Cosine mean'),
        ('baseline', 'Baseline similarity')
    ]:
        series = defaultdict(list)
        for r in rows:
            series[r['model']].append((r['step'], r[key]))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in sorted(series, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            pts = sorted(series[m])
            c = f'C{MODEL_ORDER.index(m)}' if m in MODEL_ORDER else 'C7'
            base = [v for s, v in pts if s == 0]
            cpt = [(s, v) for s, v in pts if s > 0]
            if base:
                ax.axhline(base[0], color=c, ls='--', lw=1.0, alpha=0.55)
            if cpt:
                ax.plot([s for s, _ in cpt], [v for _, v in cpt], marker='o', ms=4, color=c, label=m)
        ax.set_xlabel('CPT step')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS / f'{key}.png', dpi=150)
        plt.close(fig)
    for p in sorted(OUT.glob('*_layers.jsonl')):
        model = p.stem.replace('_layers', '')
        lrows = read_jsonl(p)
        steps = sorted({r['step'] for r in lrows})
        distinct_palette = [
            '#1f77b4',
            '#ff7f0e',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#333333',
            '#bcbd22',
            '#17becf'
        ]
        for key, label in [
            ('cosine_gap', 'Cosine gap'),
            ('p_at_1_e2x', 'P@1 eng to xho'),
            ('cosine_mean', 'Cosine mean'),
            ('baseline', 'Baseline similarity')
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            high_step_idx = 0
            for step in steps:
                pts = sorted(((r['layer'], r[key]) for r in lrows if r['step'] == step))
                if step == 0:
                    color, ls, alpha, label_str = ('black', '--', 1.0, 'Base (0)')
                elif 100 <= step <= 1000:
                    color, ls, alpha, label_str = ('#b3b3b3', '-', 0.6, 'Steps 100-1000' if step == 100 else None)
                else:
                    color = distinct_palette[high_step_idx % len(distinct_palette)]
                    ls, alpha, label_str = ('-', 1.0, f'Step {step}')
                    high_step_idx += 1
                if label_str or step > 1000:
                    ax.plot(
                        [l for l, _ in pts],
                        [v for _, v in pts],
                        marker='o',
                        ms=3,
                        color=color,
                        ls=ls,
                        alpha=alpha,
                        label=label_str
                    )
                else:
                    ax.plot(
                        [l for l, _ in pts],
                        [v for _, v in pts],
                        marker='o',
                        ms=3,
                        color=color,
                        ls=ls,
                        alpha=alpha
                    )
            ax.set_xlabel('Encoder layer')
            ax.set_ylabel(label)
            ax.set_title(f'{label} by layer for {model}')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
            fig.tight_layout()
            fig.savefig(PLOTS / f'{model}_{key}.png', dpi=150)
            plt.close(fig)


def main() -> None:
    """Plot saved alignment results without downloading checkpoints."""
    parser = argparse.ArgumentParser(description="Plot crosslingual alignment")
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.input_dir)


if __name__ == "__main__":
    main()

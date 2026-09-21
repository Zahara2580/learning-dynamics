"""Plot saved POS layer profiles, checkpoint trajectories and pooling comparisons."""

# %pip -q install pandas matplotlib

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

POOLINGS = ["mean", "first", "last"]


def plot_results(run_dir: Path) -> None:
    """Draw figures and export scores from a completed POS run directory."""
    RUN_DIR = run_dir
    RUN_MODELS = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and any(p.glob("step_*")))
    rows = []
    for path in sorted(RUN_DIR.glob('*/step_*/pooling_*/layer_*/metrics.json')):
        r = json.loads(path.read_text())
        for split in ['validation', 'test']:
            rows.append(dict(
                model=r['model'],
                step=r['step'],
                layer=r['layer'],
                pooling=r['pooling'],
                split=split,
                **{k: r[split][k] for k in ['accuracy', 'macro_f1_all_17', 'macro_f1_present']}
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No completed pooling runs found.')
    df.to_csv(RUN_DIR / 'scores.csv', index=False)
    majority = json.loads((RUN_DIR / 'majority.json').read_text())['scores']['test']
    for name in RUN_MODELS:
        selected = df[(df.model == name) & (df.split == 'test')]
        for metric in ['accuracy', 'macro_f1_present']:
            fig, axes = plt.subplots(len(POOLINGS), 2, figsize=(12, 4 * len(POOLINGS)), squeeze=False)
            for row, pool in enumerate(POOLINGS):
                pool_data = selected[selected.pooling == pool]
                for step, group in pool_data.groupby('step'):
                    group = group.sort_values('layer')
                    axes[row, 0].plot(group.layer, group[metric], label=f'CPT {step}')
                for layer, group in pool_data.groupby('layer'):
                    group = group.sort_values('step')
                    axes[row, 1].plot(group.step, group[metric], alpha=0.6)
                for ax in axes[row]:
                    ax.axhline(majority[metric], color='black', linestyle=':')
                    ax.set_ylabel(metric)
                    ax.set_title(pool)
                    ax.grid(alpha=0.2)
                axes[row, 0].set_xlabel('Encoder layer')
                axes[row, 1].set_xlabel('CPT step')
            axes[0, 0].legend(fontsize=7, ncol=4)
            fig.suptitle(name)
            fig.tight_layout()
            fig.savefig(RUN_DIR / f'{name}_{metric}.png', dpi=160)
            plt.close(fig)
        steps = sorted(selected.step.unique())
        ncols = 4
        nrows = (len(steps) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), squeeze=False)
        for ax, step in zip(axes.flat, steps):
            for pool in POOLINGS:
                group = selected[(selected.step == step) & (selected.pooling == pool)].sort_values('layer')
                ax.plot(group.layer, group.accuracy, label=pool)
            ax.set_title(f'CPT {step}')
            ax.set_xlabel('Encoder layer')
            ax.set_ylabel('Accuracy')
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(steps):]:
            ax.set_visible(False)
        axes.flat[0].legend()
        fig.suptitle(f'{name}: pooling comparison')
        fig.tight_layout()
        fig.savefig(RUN_DIR / f'{name}_pooling_comparison.png', dpi=160)
        plt.close(fig)


def main() -> None:
    """Plot an existing POS run without loading model weights."""
    parser = argparse.ArgumentParser(description="Plot POS probe results")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing manifest.json and model results."
    )
    args = parser.parse_args()
    plot_results(args.run_dir)


if __name__ == "__main__":
    main()

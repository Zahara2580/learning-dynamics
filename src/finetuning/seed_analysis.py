"""
Analysis for the byt5 D2T seed-repeat experiment.

Question: does CPT checkpoint 5,000 really do worse than base under
limited finetuning data, and does that disadvantage shrink with full data,
or is it finetuning randomness?

Reads seed_experiment/{arm}_seed{N}/results.jsonl, writes raw and summary
CSVs plus plots. Only the FINAL-EPOCH model is used, matching the paper.

Every quantity here is descriptive. Three seeds is a robustness check, not
a significance test, and nothing below computes a confidence interval.

    uv run python3 -m src.finetuning.seed_analysis
    uv run python3 -m src.finetuning.seed_analysis --arm d2t_1000
"""

import argparse
import csv
import json
import statistics
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = {"d2t_1000": "1,000 examples", "d2t_full": "full 3,859 examples"}
SEEDS = [42, 123, 456]
STEPS = [0, 5000, 10000]
METRICS = [("chrf", "chrF"), ("bleu", "BLEU")]
STEP_LABEL = {0: "base", 5000: "cpt-5k", 10000: "cpt-10k"}
SELECTION = "last_epoch"


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Seed-repeat analysis for byt5 D2T.")
    p.add_argument("--results-root", default="seed_experiment")
    p.add_argument("--out-dir", default=None, help="Default: {results-root}/analysis")
    p.add_argument("--arm", default=None, help="Only load this arm (partial run).")
    p.add_argument("--model", default="byt5")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--steps", type=int, nargs="+", default=STEPS)
    return p.parse_args()


def load_runs(root: Path, arms, seeds, model, steps) -> list[dict]:
    """One record per completed run. Missing files are reported, not fatal."""
    runs, missing = [], []
    for arm in arms:
        for seed in seeds:
            path = root / f"{arm}_seed{seed}" / "results.jsonl"
            if not path.exists():
                missing.append(str(path))
                continue
            rows = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
            found = set()
            for r in rows:
                if (r.get("selection") != SELECTION or r.get("model") != model
                        or r.get("seed") != seed or r.get("ckpt_step") not in steps):
                    continue
                found.add(r["ckpt_step"])
                runs.append({
                    "arm": arm, "seed": seed, "step": r["ckpt_step"],
                    "step_label": STEP_LABEL.get(r["ckpt_step"], str(r["ckpt_step"])),
                    "chrf": r["metrics"]["chrf"], "bleu": r["metrics"]["bleu"],
                    "chrf_pp": r["metrics"].get("chrf_pp"), "ter": r["metrics"].get("ter"),
                    "n_train_examples": r.get("n_train_examples"),
                    "checkpoint_path": r.get("checkpoint_path"),
                    "config_hash": r.get("config_hash"),
                    "best_epoch": r.get("best_epoch"), "n_epochs": r.get("n_epochs"),
                    "best_is_last": r.get("best_is_last"),
                    "train_runtime_s": r.get("train_runtime_s"),
                    "total_runtime_s": r.get("total_runtime_s"),
                    "val_loss_per_epoch": r.get("val_loss_per_epoch"),
                    "train_loss_per_epoch": r.get("train_loss_per_epoch"),
                    "sacrebleu_chrf": (r.get("sacrebleu_signatures") or {}).get("chrf"),
                    "sacrebleu_bleu": (r.get("sacrebleu_signatures") or {}).get("bleu"),
                    "timestamp": r.get("timestamp"),
                })
            for s in steps:
                if s not in found:
                    missing.append(f"{path} :: step {s} absent")
    return runs, missing


def write_raw(runs, out: Path) -> None:
    cols = ["arm", "seed", "step", "step_label", "chrf", "bleu", "chrf_pp", "ter",
            "n_train_examples", "n_epochs", "best_epoch", "best_is_last",
            "train_runtime_s", "total_runtime_s", "config_hash", "checkpoint_path",
            "sacrebleu_chrf", "sacrebleu_bleu", "timestamp"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in sorted(runs, key=lambda r: (r["arm"], r["step"], r["seed"])):
            wr.writerow(r)


def write_summary(runs, out: Path) -> list[dict]:
    """Mean and SAMPLE standard deviation across seeds, per condition."""
    by = defaultdict(list)
    for r in runs:
        by[(r["arm"], r["step"])].append(r)
    rows = []
    for (arm, step), group in sorted(by.items()):
        row = {"arm": arm, "step": step, "step_label": STEP_LABEL.get(step, str(step)),
               "n_seeds": len(group),
               "seeds": " ".join(str(s) for s in sorted(x["seed"] for x in group))}
        for key, _ in METRICS:
            vals = [x[key] for x in group]
            row[f"{key}_mean"] = round(statistics.mean(vals), 4)
            # sample sd (n-1); undefined with a single seed
            row[f"{key}_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else ""
            row[f"{key}_min"] = round(min(vals), 4)
            row[f"{key}_max"] = round(max(vals), 4)
        rows.append(row)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["arm"])
        wr.writeheader()
        wr.writerows(rows)
    return rows


def write_deltas(runs, out_delta: Path, out_did: Path):
    """checkpoint-minus-base per seed, then the change in that gap with data.

    delta      = score(step) - score(base), within one arm and seed.
                 negative = that CPT checkpoint is worse than base there.
    did        = delta(full) - delta(1,000), within one seed.
                 positive = the checkpoint's disadvantage SHRANK with full data.
    """
    idx = {(r["arm"], r["seed"], r["step"]): r for r in runs}
    deltas, dids = [], []

    for arm in ARMS:
        for seed in sorted({r["seed"] for r in runs}):
            base = idx.get((arm, seed, 0))
            if not base:
                continue
            for step in (5000, 10000):
                cur = idx.get((arm, seed, step))
                if not cur:
                    continue
                row = {"arm": arm, "seed": seed, "step": step,
                       "step_label": STEP_LABEL[step]}
                for key, _ in METRICS:
                    row[f"{key}_base"] = round(base[key], 4)
                    row[f"{key}_ckpt"] = round(cur[key], 4)
                    row[f"{key}_delta"] = round(cur[key] - base[key], 4)
                deltas.append(row)

    dmap = {(r["arm"], r["seed"], r["step"]): r for r in deltas}
    for seed in sorted({r["seed"] for r in deltas}):
        for step in (5000, 10000):
            lo = dmap.get(("d2t_1000", seed, step))
            hi = dmap.get(("d2t_full", seed, step))
            if not lo or not hi:
                continue
            row = {"seed": seed, "step": step, "step_label": STEP_LABEL[step]}
            for key, _ in METRICS:
                row[f"{key}_delta_1000"] = lo[f"{key}_delta"]
                row[f"{key}_delta_full"] = hi[f"{key}_delta"]
                row[f"{key}_did"] = round(hi[f"{key}_delta"] - lo[f"{key}_delta"], 4)
            dids.append(row)

    for path, rows in ((out_delta, deltas), (out_did, dids)):
        if not rows:
            continue
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return deltas, dids


def plot_scores(runs, out: Path) -> None:
    """Every seed as a point, the mean as a bar. Spread is across seeds."""
    arms = [a for a in ARMS if any(r["arm"] == a for r in runs)]
    fig, axes = plt.subplots(len(METRICS), len(arms), squeeze=False,
                             figsize=(4.8 * len(arms), 3.4 * len(METRICS)))
    for i, (key, mlab) in enumerate(METRICS):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            steps = sorted({r["step"] for r in runs if r["arm"] == arm})
            xs = range(len(steps))
            means = []
            for x, step in zip(xs, steps):
                vals = [r[key] for r in runs if r["arm"] == arm and r["step"] == step]
                means.append(statistics.mean(vals))
                ax.plot([x] * len(vals), vals, "o", ms=6, color="#0072B2",
                        alpha=0.75, zorder=3)
            ax.plot(list(xs), means, "_", ms=34, mew=2.5, color="#D55E00", zorder=2)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([STEP_LABEL.get(s, str(s)) for s in steps])
            ax.grid(alpha=0.3, axis="y")
            if i == 0:
                ax.set_title(ARMS[arm], fontsize=12)
            if j == 0:
                ax.set_ylabel(mlab)
    fig.suptitle("byt5 D2T, final-epoch score: each seed and the mean", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_deltas(deltas, out: Path) -> None:
    """checkpoint-minus-base, every seed shown. Zero line = base model."""
    arms = [a for a in ARMS if any(d["arm"] == a for d in deltas)]
    steps = sorted({d["step"] for d in deltas})
    fig, axes = plt.subplots(1, len(METRICS), squeeze=False,
                             figsize=(5.4 * len(METRICS), 3.8))
    for i, (key, mlab) in enumerate(METRICS):
        ax = axes[0][i]
        labels, pos = [], 0
        for step in steps:
            for arm in arms:
                vals = [d[f"{key}_delta"] for d in deltas
                        if d["arm"] == arm and d["step"] == step]
                if not vals:
                    continue
                ax.plot([pos] * len(vals), vals, "o", ms=6, color="#0072B2",
                        alpha=0.75, zorder=3)
                ax.plot([pos], [statistics.mean(vals)], "_", ms=30, mew=2.5,
                        color="#D55E00", zorder=2)
                labels.append(f"{STEP_LABEL[step]}\n{ARMS[arm].split()[0]}")
                pos += 1
        ax.axhline(0, color="0.35", lw=1.2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"{mlab}: checkpoint minus base")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Checkpoint minus base, per seed", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    root = Path(a.results_root)
    out_dir = Path(a.out_dir) if a.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.arm] if a.arm else list(ARMS)

    runs, missing = load_runs(root, arms, a.seeds, a.model, a.steps)
    print(f"loaded {len(runs)} runs from {root}")
    if missing:
        print(f"{len(missing)} expected run(s) not present:")
        for m in missing:
            print(f"  MISSING {m}")
    if not runs:
        raise SystemExit("no runs found; nothing to analyse")

    hashes = defaultdict(set)
    for r in runs:
        hashes[r["arm"]].add(r["config_hash"])
    for arm, hs in hashes.items():
        flag = "" if len(hs) == 1 else "   <-- PROTOCOL DRIFT, rows are not comparable"
        print(f"config_hash {arm}: {sorted(hs)}{flag}")

    write_raw(runs, out_dir / "raw_runs.csv")
    summary = write_summary(runs, out_dir / "summary.csv")
    deltas, dids = write_deltas(runs, out_dir / "deltas.csv",
                                out_dir / "difference_in_differences.csv")

    print(f"\n{'arm':10} {'ckpt':8} {'n':>2}  "
          + "  ".join(f"{lab+' mean':>10} {'sd':>6}" for _, lab in METRICS))
    for row in summary:
        cells = "  ".join(f"{row[k+'_mean']:>10.2f} "
                          f"{(row[k+'_sd'] if row[k+'_sd'] != '' else float('nan')):>6.2f}"
                          for k, _ in METRICS)
        print(f"{row['arm']:10} {row['step_label']:8} {row['n_seeds']:>2}  {cells}")

    if deltas:
        print(f"\ncheckpoint minus base (negative = worse than base)")
        print(f"{'arm':10} {'ckpt':8} {'seed':>5}  "
              + "  ".join(f"{lab:>8}" for _, lab in METRICS))
        for d in deltas:
            print(f"{d['arm']:10} {d['step_label']:8} {d['seed']:>5}  "
                  + "  ".join(f"{d[k+'_delta']:>+8.2f}" for k, _ in METRICS))

    if dids:
        print(f"\nchange in that gap with full data "
              f"(positive = disadvantage shrank)")
        print(f"{'ckpt':8} {'seed':>5}  "
              + "  ".join(f"{lab:>8}" for _, lab in METRICS))
        for d in dids:
            print(f"{d['step_label']:8} {d['seed']:>5}  "
                  + "  ".join(f"{d[k+'_did']:>+8.2f}" for k, _ in METRICS))
        for step in sorted({d["step"] for d in dids}):
            sel = [d for d in dids if d["step"] == step]
            print(f"{STEP_LABEL[step]:8} {'mean':>5}  "
                  + "  ".join(f"{statistics.mean([d[k+'_did'] for d in sel]):>+8.2f}"
                              for k, _ in METRICS))
    else:
        print("\n(no difference-in-differences: needs both arms for the same seed)")

    plot_scores(runs, out_dir / "scores_by_seed.png")
    if deltas:
        plot_deltas(deltas, out_dir / "deltas_by_seed.png")
    print(f"\nwrote CSVs and plots to {out_dir}")


if __name__ == "__main__":
    main()

"""
Readable tables from t2x_slot_metrics output: one table per
(arm, model, selection), checkpoints in step order, deltas against the
un-adapted base, and rank trends - same shape as summarize_results.py.

Usage:
    uv run python3 -m scripts.diagnostics.summarize_slot_metrics
    uv run python3 -m scripts.diagnostics.summarize_slot_metrics \
        --scores results/t2x_slots/scores_firstref.jsonl --arm d2t-1000
"""

import argparse
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Summarise T2X slot metrics.")
    parser.add_argument("--scores", type=str, default="results/t2x_slots/scores.jsonl")
    parser.add_argument("--arm", type=str, default=None, help="Filter to one arm.")
    parser.add_argument("--model", type=str, default=None, help="Filter to one model.")
    parser.add_argument("--selection", type=str, default=None,
                        help="Filter to best_epoch / last_epoch.")
    return parser.parse_args()


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation; no scipy dependency."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    args = parse_args()
    path = Path(args.scores)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]

    settings = {(r.get("gold_ref"), r.get("per_triple"), r.get("lowercase")) for r in rows}
    print(f"{len(rows)} rows from {path}")
    print(f"settings: {settings if len(settings) > 1 else list(settings)[0]}"
          f"   (gold_ref, per_triple, lowercase)")

    groups = defaultdict(list)
    for r in rows:
        if args.arm and r["arm"] != args.arm:
            continue
        if args.model and r["model"] != args.model:
            continue
        if args.selection and r.get("selection") != args.selection:
            continue
        groups[(r["arm"], r["model"], r.get("selection"))].append(r)

    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]))):
        arm, model, selection = key
        rs = sorted(groups[key], key=lambda r: r["step"])
        base = next((r for r in rs if r["step"] == 0), None)

        print("\n" + "=" * 96)
        print(f"{arm}  /  {model}  /  {selection or 'single-selection'}      "
              f"{len(rs)} checkpoints")
        print("=" * 96)
        print(f"{'step':>7} {'subj P':>8} {'subj R':>8} {'subj F1':>8} {'d_subj':>8} "
              f"{'obj P':>8} {'obj R':>8} {'obj F1':>8} {'d_obj':>8}")

        for r in rs:
            s, o = r["subject"], r["object"]
            ds = s["f1"] - base["subject"]["f1"] if base else 0.0
            do = o["f1"] - base["object"]["f1"] if base else 0.0
            tag = "  <- base" if r["step"] == 0 else ""
            print(f"{r['step']:>7} {s['p']:>8.2f} {s['r']:>8.2f} {s['f1']:>8.2f} {ds:>+8.2f} "
                  f"{o['p']:>8.2f} {o['r']:>8.2f} {o['f1']:>8.2f} {do:>+8.2f}{tag}")

        cpt = [r for r in rs if r["step"] > 0]
        if len(cpt) >= 3:
            steps = [float(r["step"]) for r in cpt]
            print("\n  trend over CPT checkpoints (excl. base):")
            for role in ("subject", "object"):
                vals = [r[role]["f1"] for r in cpt]
                rho = spearman(steps, vals)
                line = (f"    {role:>7} F1 vs step : rho={rho:+.3f}   "
                        f"range {min(vals):.2f}..{max(vals):.2f} "
                        f"(spread {max(vals) - min(vals):.2f})")
                if base:
                    beat = sum(1 for v in vals if v > base[role]["f1"])
                    line += f"   beat base: {beat}/{len(vals)}"
                print(line)
        if base:
            print(f"  gold copy rate: subject {base['subject']['gold_copy_rate']:.3f}, "
                  f"object {base['object']['gold_copy_rate']:.3f}  "
                  f"(n={base['subject']['n']} slots)")


if __name__ == "__main__":
    main()

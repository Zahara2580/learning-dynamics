"""
Inspect the WMT22 MT training selection: laser_score distribution of the
kept top-N, and sample pairs at the top / median / bottom of the kept
set, to judge whether the training data is any good.

Usage:
    uv run python3 -m src.finetuning.inspect_mt_train --config configs/finetune/mt.yaml
"""

import argparse
from argparse import Namespace

import numpy as np
from datasets import load_dataset

from src.finetuning.config import FinetuneConfig
from src.finetuning.data_mt import (
    WMT22_CONFIG,
    WMT22_DATASET,
    rank_by_laser,
    select_top_n_unique,
)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Inspect the WMT22 top-N MT training selection.")
    parser.add_argument("--config", type=str, required=True, help="MT finetune config YAML.")
    parser.add_argument("--n-floor", type=int, default=8, help="How many lowest-scoring kept pairs to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)

    raw = load_dataset(WMT22_DATASET, WMT22_CONFIG, split="train")

    # Exactly the selection load_mt_train uses, so the inspector always
    # reflects the real training set; the kept rows carry laser_score.
    rows, n_dup = select_top_n_unique(rank_by_laser(raw), cfg.n_train_pairs)
    kept = [(r["laser_score"], r["translation"]["eng"].strip(), r["translation"]["xho"].strip())
            for r in rows]

    sc = np.array([k[0] for k in kept])
    print("=" * 78)
    print(f"WMT22 eng-xho: kept top {len(kept)} unique pairs by laser_score")
    print(f"  score  min={sc.min():.3f}  p25={np.percentile(sc, 25):.3f}  "
          f"median={np.percentile(sc, 50):.3f}  p75={np.percentile(sc, 75):.3f}  max={sc.max():.3f}")
    print(f"  the cutoff (lowest kept score) is {sc.min():.3f} - pairs below this were not used")
    print("=" * 78)

    labels = {"HIGHEST score": 0, "MEDIAN score": len(kept) // 2, "LOWEST kept score": len(kept) - 1}
    for label, j in labels.items():
        score, en, xh = kept[j]
        print(f"\n[{label}]  laser={score:.3f}")
        print(f"  EN: {en[:220]}")
        print(f"  XH: {xh[:220]}")

    print(f"\n{'-' * 78}\nThe {args.n_floor} lowest-scoring kept pairs (the quality floor of the training set):")
    for score, en, xh in kept[-args.n_floor:]:
        print(f"\n  [{score:.3f}] EN: {en[:160]}")
        print(f"          XH: {xh[:160]}")


if __name__ == "__main__":
    main()

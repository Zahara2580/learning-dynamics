"""
Byte/token length coverage for MT targets vs the truncation cap.

max_target_length (and max_new_tokens) count BYTES for byt5/nguni-byt5
but TOKENS for t5. If isiXhosa targets exceed the cap, the byte models
train on truncated targets and cannot reach long references while t5 has
headroom. The cap may be raised ONCE, before any real MT run.

Usage:
    uv run python3 -m src.finetuning.check_lengths --config configs/finetune/mt.yaml
"""

import argparse
import logging
from argparse import Namespace

import numpy as np
from transformers import AutoTokenizer

from src.finetuning.config import FinetuneConfig
from src.finetuning.data_mt import (
    FLORES_TEST_SPLIT,
    FLORES_VALIDATION_SPLIT,
    load_flores_split,
    load_mt_train,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Byte model first (its tokens are bytes, so most at risk), subword anchor second.
TOKENIZERS = ["google/byt5-large", "google-t5/t5-large"]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="MT target length coverage vs the cap.")
    parser.add_argument("--config", type=str, required=True, help="MT finetune config YAML.")
    return parser.parse_args()


def summarise(lengths: list[int], cap: int) -> dict:
    """Percentiles, max, and over-cap count for one length distribution."""
    arr = np.asarray(lengths)
    return {
        "n": len(arr),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "max": int(arr.max()),
        "over": int((arr > cap).sum()),
    }


def print_row(label: str, s: dict, cap: int) -> None:
    print(f"  {label:16} n={s['n']:6} p50={s['p50']:6.0f} p90={s['p90']:6.0f} "
          f"p95={s['p95']:6.0f} p99={s['p99']:6.0f} max={s['max']:6} over_{cap}={s['over']}")


def main() -> None:
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)
    cap = cfg.max_target_length

    logger.info("loading MT target corpora (downloads/sorts WMT22 on first run)...")
    _, train_targets = load_mt_train(cfg)  # reuses the real top-N selection
    _, dev_refs = load_flores_split(cfg.data_dir, FLORES_VALIDATION_SPLIT)
    _, test_refs = load_flores_split(cfg.data_dir, FLORES_TEST_SPLIT)
    corpora = {
        "wmt22_train": train_targets,
        "flores_dev": [r[0] for r in dev_refs],
        "flores_devtest": [r[0] for r in test_refs],
    }

    print("\n" + "=" * 78)
    print(f"MT target length coverage   (cap = max_target_length = {cap})")
    print("=" * 78)

    print("\nUTF-8 byte length (tokenizer-independent):")
    for name, targets in corpora.items():
        print_row(name, summarise([len(t.encode("utf-8")) for t in targets], cap), cap)

    verdicts = []
    for tok_name in TOKENIZERS:
        print(f"\n{tok_name} token length:")
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        worst_p99, total_over, total_n = 0, 0, 0
        for name, targets in corpora.items():
            s = summarise([len(tokenizer.encode(t)) for t in targets], cap)
            print_row(name, s, cap)
            worst_p99 = max(worst_p99, s["p99"])
            total_over += s["over"]
            total_n += s["n"]
        if worst_p99 <= cap:
            verdicts.append(f"OK: p99 <= {cap} for {tok_name}")
        else:
            verdicts.append(
                f"RAISE CAP: {total_over} of {total_n} targets exceed {cap} for {tok_name}")

    print("\n" + "=" * 78)
    for verdict in verdicts:
        print("  " + verdict)
    print("=" * 78)


if __name__ == "__main__":
    main()

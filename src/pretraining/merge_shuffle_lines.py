"""
Seeded merge + shuffle of line files (bilingual CPT corpus).

Usage:
    uv run python3 -m src.pretraining.merge_shuffle_lines \
        --inputs train.xh train_eng_sample.txt --output train_bilingual.txt
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded merge+shuffle of line files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = []
    for f in args.inputs:
        chunk = Path(f).read_text(encoding="utf-8").splitlines()
        print(f"{f}: {len(chunk):,} lines")
        lines.extend(chunk)
    random.Random(args.seed).shuffle(lines)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(lines):,} lines")


if __name__ == "__main__":
    main()

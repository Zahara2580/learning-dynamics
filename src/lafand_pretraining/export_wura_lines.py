"""
Export the raw WURA corpus to plain text, one paragraph per line - the
input format the lafand-mt (AfriByT5) pretraining pipeline consumes.

The lafand preprocess step (lafand_preprocess.py, ported from
mt5_byt5_pre_training/process.py) reads a text file with .readlines()
and treats every line as one training example. WURA is document-level
(headline + content), so this script splits each document into paragraph
lines: headline as its own line, then content split on newlines, empty
and near-empty lines dropped.

This step is model-agnostic (no tokenizer) - run once per split and the
output is shared by all three models' preprocessing:

    uv run python3 -m src.lafand_pretraining.export_wura_lines --input /scratch/rmdrak003/data/corpus/xho --split train --output /scratch/rmdrak003/data/lafand/lines/train.xh
    uv run python3 -m src.lafand_pretraining.export_wura_lines --input /scratch/rmdrak003/data/corpus/xho --split validation --output /scratch/rmdrak003/data/lafand/lines/dev.xh
"""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Export raw WURA to one-paragraph-per-line text.")
    parser.add_argument("--input", type=str, required=True, help="Raw corpus dir (as saved by download_corpus.py).")
    parser.add_argument("--split", type=str, default="train", help="Which split of the raw DatasetDict to export.")
    parser.add_argument("--output", type=str, required=True, help="Output text file path.")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=10,
        help="Drop paragraph lines shorter than this many characters. DEVIATION "
             "from lafand (which reads all lines unfiltered), added because: "
             "(1) empty lines crash their own dataset loader (util.py asserts "
             "min line length > 0); (2) lines under ~10 chars tokenize so short "
             "that int(len*0.15) rounds to zero masked positions, yielding "
             "degenerate examples with nothing to predict - 10 chars is roughly "
             "the smallest line that guarantees at least one masked byte token. "
             "Measured cost on WURA xho: ~4.8%% of lines but only ~0.12%% of "
             "characters (dropped lines average ~4 chars).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading raw corpus from {args.input} (split={args.split})...")
    corpus = load_from_disk(args.input)[args.split]
    logger.info(f"{len(corpus):,} documents.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_lines, n_dropped, total_chars, dropped_chars = 0, 0, 0, 0
    with open(output_path, "w") as f:
        for doc in corpus:
            text = f"{doc['headline']}\n{doc['content']}"
            for line in text.split("\n"):
                line = line.strip()
                if len(line) < args.min_chars:
                    n_dropped += 1
                    dropped_chars += len(line)
                    continue
                f.write(line + "\n")
                n_lines += 1
                total_chars += len(line)

    char_loss = dropped_chars / max(total_chars + dropped_chars, 1)
    logger.info(
        f"Wrote {n_lines:,} paragraph lines to {output_path} "
        f"(mean length {total_chars / max(n_lines, 1):.0f} chars). "
        f"min-chars filter dropped {n_dropped:,} lines = {dropped_chars:,} chars "
        f"({char_loss:.4%} of total characters)."
    )


if __name__ == "__main__":
    main()

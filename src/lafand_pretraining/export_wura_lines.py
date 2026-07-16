"""
Export the passage-level WURA corpus to plain text, one passage per
line - the input format the lafand-mt (AfriByT5) pretraining pipeline
consumes.

Per supervisor guidance ("process the data examples (passages) as is"),
WURA's own pre-segmented passages are used directly, with no manual
splitting: one passage = one training example. The lafand preprocess
step (lafand_preprocess.py) reads this file with .readlines() and
treats every line as one example; batching, truncation and padding are
left to the HuggingFace collator/Trainer downstream.

The only filtering is a minimum length (empty/near-empty lines crash
the lafand data loader and produce degenerate masking examples).

    uv run python3 -m src.lafand_pretraining.export_wura_lines --input /scratch/rmdrak003/data/corpus/xho-passage --split train --output /scratch/rmdrak003/data/lafand/lines-passage/train.xh
    uv run python3 -m src.lafand_pretraining.export_wura_lines --input /scratch/rmdrak003/data/corpus/xho-passage --split validation --output /scratch/rmdrak003/data/lafand/lines-passage/dev.xh
"""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Export passage-level WURA to one-passage-per-line text.")
    parser.add_argument("--input", type=str, required=True,
                        help="Passage-level corpus dir (download_corpus.py --level passage).")
    parser.add_argument("--split", type=str, default="train", help="Which split of the DatasetDict to export.")
    parser.add_argument("--output", type=str, required=True, help="Output text file path.")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=10,
        help="Drop passages shorter than this many characters. Empty lines "
             "crash the lafand data loader (util.py asserts min line length "
             "> 0), and <10-char lines tokenize so short that int(len*0.15) "
             "rounds to zero masked positions (degenerate examples).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading passage-level corpus from {args.input} (split={args.split})...")
    corpus = load_from_disk(args.input)[args.split]
    logger.info(f"{len(corpus):,} passages.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_lines, n_dropped, total_chars, dropped_chars = 0, 0, 0, 0
    with open(output_path, "w") as f:
        for example in corpus:
            # One passage = one line, as-is. Any internal newlines are
            # flattened to spaces so one passage stays one line.
            line = " ".join(example["text"].split()).strip()
            if len(line) < args.min_chars:
                n_dropped += 1
                dropped_chars += len(line)
                continue
            f.write(line + "\n")
            n_lines += 1
            total_chars += len(line)

    char_loss = dropped_chars / max(total_chars + dropped_chars, 1)
    logger.info(
        f"Wrote {n_lines:,} passages to {output_path} "
        f"(mean length {total_chars / max(n_lines, 1):.0f} chars). "
        f"min-chars filter dropped {n_dropped:,} passages = {dropped_chars:,} chars "
        f"({char_loss:.4%} of total characters)."
    )


if __name__ == "__main__":
    main()

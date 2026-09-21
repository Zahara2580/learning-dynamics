"""Export WURA passages as one passage per line."""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

from datasets import load_from_disk
import random as _random
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """Parse corpus export options."""
    parser = argparse.ArgumentParser(
        description='Export passage level WURA to one passage per line text',
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Prepared passage-level corpus directory.',
    )
    parser.add_argument('--split', type=str, default='train', help='Dataset split to export.')
    parser.add_argument('--output', type=str, required=True, help='Output text file.')
    parser.add_argument(
        '--sample-n',
        type=int,
        default=None,
        help='Randomly sample this many passages, defaults to all.',
    )
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument(
        '--min-chars',
        type=int,
        default=1,
        help='Minimum passage length; defaults to keeping nonempty passages.',
    )
    return parser.parse_args()


def main() -> None:
    """Export the selected corpus split to a text file."""
    args = parse_args()

    logger.info(f"Loading passage level corpus from {args.input} (split={args.split})")
    corpus = load_from_disk(args.input)[args.split]
    logger.info(f"{len(corpus):} passages.")

    if args.sample_n is not None:
        if args.sample_n > len(corpus):
            raise ValueError(f"--sample-n {args.sample_n} > corpus size {len(corpus)}")

        idx = sorted(_random.Random(args.sample_seed).sample(range(len(corpus)), args.sample_n))
        corpus = corpus.select(idx)
        logger.info(f"Sampled {len(corpus):} passages (seed {args.sample_seed}).")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    with open(output_path, "w") as f:
        for example in corpus:
            line = " ".join(example["text"].split()).strip()
            if len(line) < args.min_chars:
                continue
            f.write(line + "\n")
            n_lines += 1

    logger.info(f"Wrote {n_lines} passages to {output_path}")


if __name__ == "__main__":
    main()

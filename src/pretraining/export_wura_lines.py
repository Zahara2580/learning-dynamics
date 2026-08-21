"""
Export the passage-level WURA corpus to plain text, one passage per
line, which is the input format the lafand-mt pipeline consumes.

WURA's own pre-segmented passages are used as-is: one passage is one
training example. lafand_preprocess.py reads the file with readlines()
and treats each line as an example. The only filtering is a minimum
length, since empty lines crash the lafand loader.

    uv run python3 -m src.pretraining.export_wura_lines --input /scratch/rmdrak003/data/corpus/xho-passage --split train --output /scratch/rmdrak003/data/lafand/lines-passage/train.xh
"""

import argparse
import json
import logging
import sys
from argparse import Namespace
from datetime import datetime, timezone
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
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Randomly sample this many passages instead of exporting all "
                             "(bilingual corpus construction: match the xho passage count).")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Drop passages shorter than this many characters. Default 1 = "
             "drop only empty lines, which crash the lafand data loader "
             "(util.py asserts min line length > 0). Content filtering is "
             "deliberately OFF (measured: a 10-char threshold dropped zero "
             "passages on both WURA splits anyway).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading passage-level corpus from {args.input} (split={args.split})...")
    corpus = load_from_disk(args.input)[args.split]
    logger.info(f"{len(corpus):,} passages.")

    if args.sample_n is not None:
        if args.sample_n > len(corpus):
            raise ValueError(f"--sample-n {args.sample_n} > corpus size {len(corpus)}")
        import random as _random
        idx = sorted(_random.Random(args.sample_seed).sample(range(len(corpus)), args.sample_n))
        corpus = corpus.select(idx)
        logger.info(f"Sampled {len(corpus):,} passages (seed {args.sample_seed}).")

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

    # Provenance beside the output. This script is run by hand, so without a
    # record here nothing downstream can say which corpus/split it consumed -
    # which is exactly how two similar exports became ambiguous once.
    # See notes/things_to_fix_next_pipeline.md A9.
    (output_path.parent / f"{output_path.name}.provenance.json").write_text(
        json.dumps({
            "input_corpus": str(Path(args.input).resolve()),
            "split": args.split,
            "min_chars": args.min_chars,
            "sample_n": args.sample_n,
            "sample_seed": args.sample_seed,
            "passages_written": n_lines,
            "passages_dropped": n_dropped,
            "characters_written": total_chars,
            "argv": sys.argv,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")

    char_loss = dropped_chars / max(total_chars + dropped_chars, 1)
    logger.info(
        f"Wrote {n_lines:,} passages to {output_path} "
        f"(mean length {total_chars / max(n_lines, 1):.0f} chars). "
        f"min-chars filter dropped {n_dropped:,} passages = {dropped_chars:,} chars "
        f"({char_loss:.4%} of total characters)."
    )


if __name__ == "__main__":
    main()

"""
Size comparison of corpus line files across every unit that matters for
the bilingual CPT decision.

Equal PASSAGES (the supervisor's stated unit) does not mean equal
TOKENS, and token parity is tokenizer-dependent: the subword tokenizer
is English-optimised, the byte models are neutral. This report puts all
the units side by side so the parity decision is made on numbers.

The last column - projected 512-token windows - is the unit that
actually determines how many training examples CPT sees.

Usage:
    uv run python3 -m scripts.diagnostics.corpus_token_stats \
        --file xho=/scratch/rmdrak003/data/lafand/lines-passage/train.xh \
        --file eng=/scratch/rmdrak003/data/lafand/lines-passage/train_eng_sample.txt
"""

import argparse
import math
from argparse import Namespace
from pathlib import Path

from transformers import AutoTokenizer

WINDOW = 512
T5_TOKENIZER = "google-t5/t5-large"


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Corpus size across units.")
    parser.add_argument("--file", action="append", required=True,
                        help="label=path, repeatable (e.g. xho=... eng=...)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N lines (fast preview).")
    return parser.parse_args()


def stats(path: Path, tokenizer, limit: int | None) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]

    chars = sum(len(l) for l in lines)
    utf8 = sum(len(l.encode("utf-8")) for l in lines)

    # byte models: tokens = utf8 bytes + eos per passage
    byte_tokens = utf8 + len(lines)
    byte_windows = sum(math.ceil((len(l.encode("utf-8")) + 1) / WINDOW) for l in lines)

    t5_tokens, t5_windows = 0, 0
    batch = 512
    for start in range(0, len(lines), batch):
        enc = tokenizer(lines[start:start + batch])["input_ids"]
        t5_tokens += sum(len(e) for e in enc)
        t5_windows += sum(math.ceil(len(e) / WINDOW) for e in enc)

    return {"passages": len(lines), "chars": chars, "utf8_bytes": utf8,
            "byte_tokens": byte_tokens, "byte_windows": byte_windows,
            "t5_tokens": t5_tokens, "t5_windows": t5_windows}


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(T5_TOKENIZER)

    results = {}
    for spec in args.file:
        label, _, path = spec.partition("=")
        print(f"reading {label}: {path}")
        results[label] = stats(Path(path), tokenizer, args.limit)

    keys = ["passages", "chars", "utf8_bytes", "byte_tokens", "byte_windows",
            "t5_tokens", "t5_windows"]
    labels = list(results)

    print(f"\n{'unit':>14}" + "".join(f"{l:>16}" for l in labels)
          + ("" if len(labels) < 2 else f"{labels[1] + '/' + labels[0]:>14}"))
    for k in keys:
        row = f"{k:>14}" + "".join(f"{results[l][k]:>16,}" for l in labels)
        if len(labels) >= 2 and results[labels[0]][k]:
            row += f"{results[labels[1]][k] / results[labels[0]][k]:>14.3f}"
        print(row)

    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        print(f"\nmean passage length: {a} {results[a]['chars']/max(1,results[a]['passages']):.0f} chars, "
              f"{b} {results[b]['chars']/max(1,results[b]['passages']):.0f} chars")
        print(f"chars/t5-token     : {a} {results[a]['chars']/max(1,results[a]['t5_tokens']):.2f}, "
              f"{b} {results[b]['chars']/max(1,results[b]['t5_tokens']):.2f}")


if __name__ == "__main__":
    main()

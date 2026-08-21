"""
Faithful port of lafand-mt mt5_byt5_pre_training/process.py: offline
span-corruption preprocessing that writes train.source/train.target
(dev.source/dev.target) files of space-separated token ids - the exact
pipeline used to continued-pretrain Nguni-ByT5 (per supervisor mandate,
this replaces our previous online DataCollatorForT5MLM approach).

Corruption: 15% of token positions sampled i.i.d. (not span-constructed),
so masked runs are geometric with mean ~1.18 tokens. Consecutive
positions merge into one run, one sentinel per run. The target is the
complement, resampled until it starts with the first sentinel, which
keeps source and target sentinel numbering aligned. One fixed corruption
per line, written to disk.

The first sentinel id is read from the tokenizer rather than hardcoded
to 258: t5 -> 32099 (descending), byt5/nguni-byt5 -> 259 (ascending).
Passages are split into --max-line-tokens (default 512) windows before
masking; longer windows can need more sentinels than the vocabulary
defines (100 for t5, 125 for byt5) and would encode garbage. 512 bounds
runs to ~65. Seeded for reproducibility.

Usage (once per model per split):
    uv run python3 -m src.pretraining.lafand_preprocess --input-text /scratch/rmdrak003/data/lafand/lines-passage/train.xh --model-config configs/models/byt5.yaml --output-dir /scratch/rmdrak003/data/lafand/byt5 --type-path train
    uv run python3 -m src.pretraining.lafand_preprocess --input-text /scratch/rmdrak003/data/lafand/lines-passage/dev.xh --model-config configs/models/byt5.yaml --output-dir /scratch/rmdrak003/data/lafand/byt5 --type-path dev
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from src.pretraining.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resampling the noise until the target starts with the first sentinel
# succeeds with p=0.85 per attempt; this cap exists only to guarantee
# termination on pathological lines and statistically never triggers.
MAX_RESAMPLE_ATTEMPTS = 1000


def _head_digest(path: str, n_bytes: int = 1 << 20) -> str:
    """sha256 of the first 1MB of the input, so a later reader can confirm the
    file on disk is still the one that was consumed."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(n_bytes)).hexdigest()


def racha_detection(lista):
    # Returns a list of lists where each sub-list contains the
    # consecutive tokens in the list (verbatim from lafand process.py).
    rachas = []
    racha = []
    for i, element in enumerate(lista):
        if (i < len(lista) - 1) and (lista[i + 1] == element + 1):
            racha.append(element)
        else:
            if len(racha) > 0:
                rachas.append(racha + [element])
            else:
                rachas.append([element])
            racha = []
    return rachas


def masking(tokenized_sentence, rachas, sentinel_ids):
    # Masks a tokenized sentence following the rachas: one sentinel per
    # racha, other positions in the racha dropped (verbatim logic; the
    # sentinel id lookup is precomputed in sentinel_ids).
    sent_token_id = 0
    enmascared = tokenized_sentence.copy()
    for racha in rachas:
        sent_id = sentinel_ids[sent_token_id]
        for i, idx in enumerate(racha):
            if i == 0:
                enmascared[idx] = sent_id
            else:
                enmascared[idx] = -100
        sent_token_id += 1
    enmascared = [t for t in enmascared if t != -100]
    return enmascared


def add_noise(tokenized_sentence, sentinel_ids, percent=0.15):
    # Takes a pre-tokenized sentence and returns masked input ids and
    # masked target ids following the T5 span-corruption scheme
    # (verbatim from lafand process.py, tokenization hoisted out so the
    # resample loop doesn't re-tokenize).
    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)),
                                       int(len(tokenized_sentence) * percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas, sentinel_ids)

    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas, sentinel_ids)

    return enmascared_input, enmascared_target


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Offline lafand-style span corruption preprocessing.")
    parser.add_argument("--input-text", type=str, required=True, help="One-example-per-line text file (from export_wura_lines.py).")
    parser.add_argument("--model-config", type=str, required=True, help="Model YAML (provides the tokenizer).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for {type_path}.source/.target files.")
    parser.add_argument("--type-path", type=str, default="train",
                        help="Output file prefix (train, dev, dev_xho, dev_eng, ...).")
    parser.add_argument("--max-line-tokens", type=int, default=512,
                        help="Window size (tokens). Each passage is tokenized then split into "
                             "consecutive windows of this length; each window is masked as its "
                             "own example. This keeps all the text (no truncation of long "
                             "passages) - the 'let HuggingFace split into equal-length "
                             "sequences' step, done here so masking sees fixed-size windows "
                             "within the sentinel-vocabulary limit.")
    parser.add_argument("--drop-last-window", action="store_true",
                        help="Drop each passage's final short window instead of keeping it. "
                             "Off by default: short trailing windows are kept (still valid "
                             "examples), so no text is discarded.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N lines (for smoke tests).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    config = ModelConfig.from_yaml(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Dynamic first-sentinel id, per the lafand README's own instruction.
    first_sentinel = tokenizer.encode("<extra_id_0>")[0]
    # Precompute the sentinel id sequence the same way lafand does
    # (tokenizer.encode(f'<extra_id_{i}>')[0]), for every extra_id the
    # tokenizer defines.
    sentinel_ids = []
    i = 0
    while i <= 200:  # t5 has 100, byt5 has 125; hard stop as a sanity bound
        encoded = tokenizer.encode(f"<extra_id_{i}>")
        # A real sentinel encodes to exactly [sentinel_id, eos]; a
        # nonexistent one gets split into many pieces (byte-encoded for
        # byt5, sentencepiece-split for t5) - that's where the vocabulary
        # of sentinels ends.
        if len(encoded) != 2:
            break
        sentinel_ids.append(encoded[0])
        i += 1
    logger.info(
        f"Model: {config.model_name_or_path} | first sentinel id: {first_sentinel} | "
        f"{len(sentinel_ids)} sentinel ids available "
        f"({'ascending' if len(sentinel_ids) > 1 and sentinel_ids[1] > sentinel_ids[0] else 'descending'})"
    )

    lines = open(args.input_text).readlines()
    if args.limit is not None:
        lines = lines[: args.limit]
    logger.info(f"Processing {len(lines):,} lines from {args.input_text}...")

    sources, targets = [], []
    n_passages, n_windows, n_skipped, total_tokens = 0, 0, 0, 0
    W = args.max_line_tokens
    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
        n_passages += 1
        tokenized = tokenizer.encode(line)

        # Split the passage into consecutive fixed-size windows; each
        # window becomes its own masked example, so long passages keep
        # all their text instead of being truncated.
        for start in range(0, len(tokenized), W):
            window = tokenized[start:start + W]
            # Drop a too-short final window (nothing to mask) or, if
            # requested, any final window shorter than W.
            if len(window) < 2 or (args.drop_last_window and len(window) < W):
                continue
            total_tokens += len(window)

            source, target = add_noise(window, sentinel_ids)
            attempts = 1
            while target[0] != first_sentinel and attempts < MAX_RESAMPLE_ATTEMPTS:
                source, target = add_noise(window, sentinel_ids)
                attempts += 1
            if target[0] != first_sentinel:
                n_skipped += 1
                continue

            sources.append(" ".join(map(str, source)).strip() + "\n")
            targets.append(" ".join(map(str, target)).strip() + "\n")
            n_windows += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.type_path}.source", "w") as f:
        f.writelines(sources)
    with open(output_dir / f"{args.type_path}.target", "w") as f:
        f.writelines(targets)

    # Provenance beside the data. Without this, "which corpus fed CPT?" is only
    # answerable by decoding token ids and comparing segment-length statistics -
    # which is what it took once, during write-up, when two similar exports of
    # the corpus existed side by side. See notes/things_to_fix_next_pipeline.md A9.
    provenance = {
        "input_text": str(Path(args.input_text).resolve()),
        "input_lines": n_passages,
        "input_sha256_first_1mb": _head_digest(args.input_text),
        "model_config": args.model_config,
        "tokenizer": config.model_name_or_path,
        "max_line_tokens": args.max_line_tokens,
        "drop_last_window": args.drop_last_window,
        "seed": args.seed,
        "noise_density": 0.15,
        "first_sentinel_id": first_sentinel,
        "n_sentinels_available": len(sentinel_ids),
        "output_examples": n_windows,
        "skipped": n_skipped,
        "argv": sys.argv,
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    prov_path = output_dir / f"{args.type_path}.provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")

    logger.info(
        f"Wrote {n_windows:,} segment-examples from {n_passages:,} passages to "
        f"{output_dir}/{args.type_path}.source/.target "
        f"(mean {total_tokens / max(n_windows, 1):.0f} tokens/segment; "
        f"{n_windows / max(n_passages, 1):.2f} segments/passage; {n_skipped:,} skipped)."
    )
    logger.info(f"Provenance written to {prov_path}")


if __name__ == "__main__":
    main()

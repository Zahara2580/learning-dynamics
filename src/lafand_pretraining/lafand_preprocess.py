"""
Faithful port of lafand-mt mt5_byt5_pre_training/process.py: offline
span-corruption preprocessing that writes train.source/train.target
(dev.source/dev.target) files of space-separated token ids - the exact
pipeline used to continued-pretrain Nguni-ByT5 (per supervisor mandate,
this replaces our previous online DataCollatorForT5MLM approach).

Mechanics preserved verbatim from the original:
  - 15% of token positions sampled i.i.d. (random.sample), NOT
    span-constructed -> masked runs are geometric, mean ~1.18 tokens
  - consecutive positions merged into one run (racha_detection), one
    sentinel token per run (masking)
  - target = complement masking; the while-loop resamples until the
    target starts with the first sentinel (i.e. position 0 unmasked),
    which keeps input/target sentinel numbering aligned
  - one fixed corruption per line, written to disk (offline masking)

Sanctioned adaptations (each per the repo's own README or necessary):
  - the hardcoded first-sentinel id 258 is computed dynamically as
    tokenizer.encode('<extra_id_0>')[0], per the README instruction
    ("change the number 258 to the first token id when using mT5"):
    t5 -> 32099 (descending), byt5/nguni-byt5 -> 259 (ascending on
    current transformers - matching what nguni-byt5 was actually
    trained with, confirmed by embedding forensics)
  - lines are pre-truncated to --max-line-tokens (default 512) BEFORE
    masking: without this, paragraphs long enough to produce more
    masked runs than there are <extra_id_*> tokens (100 for t5, 125
    for byt5) would silently encode nonexistent sentinel strings as
    garbage tokens. 512 bounds runs to ~65, well inside both limits.
  - random.seed(--seed) for reproducibility (original was unseeded)
  - sentinel ids are cached per index instead of re-encoded per run
    (identical values, just faster)

Usage (once per model per split):
    uv run python3 -m src.lafand_pretraining.lafand_preprocess --input-text /scratch/rmdrak003/data/lafand/lines/train.xh --model-config configs/models/byt5.yaml --output-dir /scratch/rmdrak003/data/lafand/byt5 --type-path train
    uv run python3 -m src.lafand_pretraining.lafand_preprocess --input-text /scratch/rmdrak003/data/lafand/lines/dev.xh --model-config configs/models/byt5.yaml --output-dir /scratch/rmdrak003/data/lafand/byt5 --type-path dev
"""

import argparse
import logging
import random
from argparse import Namespace
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
    parser.add_argument("--type-path", type=str, default="train", choices=["train", "dev"], help="Output file prefix.")
    parser.add_argument("--max-line-tokens", type=int, default=512,
                        help="Pre-truncate tokenized lines to this many tokens before masking, "
                             "bounding masked-run count below the sentinel vocabulary limit.")
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
    n_truncated, n_skipped, total_tokens = 0, 0, 0
    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
        tokenized = tokenizer.encode(line)
        if len(tokenized) > args.max_line_tokens:
            tokenized = tokenized[: args.max_line_tokens]
            n_truncated += 1
        if len(tokenized) < 2:
            n_skipped += 1
            continue
        total_tokens += len(tokenized)

        source, target = add_noise(tokenized, sentinel_ids)
        attempts = 1
        while target[0] != first_sentinel and attempts < MAX_RESAMPLE_ATTEMPTS:
            source, target = add_noise(tokenized, sentinel_ids)
            attempts += 1
        if target[0] != first_sentinel:
            n_skipped += 1
            continue

        sources.append(" ".join(map(str, source)).strip() + "\n")
        targets.append(" ".join(map(str, target)).strip() + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.type_path}.source", "w") as f:
        f.writelines(sources)
    with open(output_dir / f"{args.type_path}.target", "w") as f:
        f.writelines(targets)

    logger.info(
        f"Wrote {len(sources):,} examples to {output_dir}/{args.type_path}.source/.target "
        f"(mean {total_tokens / max(len(sources), 1):.0f} tokens/line; "
        f"{n_truncated:,} lines truncated to {args.max_line_tokens} tokens; {n_skipped:,} skipped)."
    )


if __name__ == "__main__":
    main()

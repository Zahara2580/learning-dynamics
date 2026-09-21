"""Prepare tokenized source and target files using span corruption."""

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


MAX_RESAMPLE_ATTEMPTS = 1000


def racha_detection(lista):
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
    idxs_2_mask = sorted(random.sample(range(len(tokenized_sentence)),
                                       int(len(tokenized_sentence) * percent)))
    rachas = racha_detection(idxs_2_mask)
    enmascared_input = masking(tokenized_sentence, rachas, sentinel_ids)

    idxs_2_mask = [idx for idx in range(len(tokenized_sentence)) if idx not in idxs_2_mask]
    rachas = racha_detection(idxs_2_mask)
    enmascared_target = masking(tokenized_sentence, rachas, sentinel_ids)

    return enmascared_input, enmascared_target


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Prepare span-corrupted training data")
    parser.add_argument(
        '--input-text',
        type=str,
        required=True,
        help='Text file with one passage per line.',
    )
    parser.add_argument('--model-config', type=str, required=True, help='Model configuration YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument(
        '--type-path',
        type=str,
        default='train',
        help='Output split name, e.g. train, dev, dev_xho or dev_eng',
    )
    parser.add_argument(
        '--max-line-tokens',
        type=int,
        default=512,
        help='Token window size before masking',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N lines')
    return parser.parse_args()


def main() -> None:
    """Tokenize passages, apply span corruption and save source-target pairs"""
    args = parse_args()
    random.seed(args.seed)

    config = ModelConfig.from_yaml(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    first_sentinel = tokenizer.encode("<extra_id_0>")[0]

    sentinel_ids = []
    i = 0
    while i <= 200:
        encoded = tokenizer.encode(f"<extra_id_{i}>")

        if len(encoded) != 2:
            break
        sentinel_ids.append(encoded[0])
        i += 1
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"First sentinel: {first_sentinel}")
    logger.info(f"Available sentinels: {len(sentinel_ids)}")

    lines = open(args.input_text).readlines()
    if args.limit is not None:
        lines = lines[: args.limit]
    logger.info(f"Processing {len(lines):} lines from {args.input_text}...")

    sources, targets = [], []
    n_passages, n_windows, n_skipped, total_tokens = 0, 0, 0, 0
    W = args.max_line_tokens
    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
        n_passages += 1
        tokenized = tokenizer.encode(line)

        for start in range(0, len(tokenized), W):
            window = tokenized[start:start + W]

            if len(window) < 2:
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

    logger.info(f"Output: {output_dir}/{args.type_path}.source/.target")
    logger.info(f"Passages: {n_passages}, examples: {n_windows}, skipped: {n_skipped}")
    logger.info(f"Mean tokens per example: {total_tokens / max(n_windows, 1):.0f}")
    logger.info(f"Examples per passage: {n_windows / max(n_passages, 1):.2f}")


if __name__ == "__main__":
    main()

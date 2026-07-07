"""
Standalone sanity-check script for DataCollatorForT5MLM.

Run directly (not via pytest) for a fast, informative check that the
collator produces correctly-shaped, sensible output before wiring it
into the full continued_pretrain.py training script.

Usage:
    uv run python3 -m src.unit_tests.test_collator --model-config configs/models/t5.yaml --input /path/to/preprocessed/wura/chunks
"""

import argparse
import logging
from argparse import Namespace

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.pretraining.collator import DataCollatorForT5MLM, compute_input_and_target_lengths
from src.pretraining.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# T5's standard span-corruption hyperparameters
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check DataCollatorForT5MLM.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed WURA chunks.")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    expanded_length, target_length = compute_input_and_target_lengths(
        input_length=config.max_seq_length,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
    )
    logger.info(
        f"max_seq_length={config.max_seq_length} -> "
        f"expanded_length={expanded_length}, target_length={target_length}"
    )

    collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
        input_length=config.max_seq_length,
        target_length=target_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    logger.info(f"Loading preprocessed chunks from {args.input}...")
    dataset = load_from_disk(args.input)
    logger.info(f"Loaded {len(dataset):,} chunks, each of length {len(dataset[0]['input_ids'])}")

    actual_chunk_length = len(dataset[0]["input_ids"])
    if actual_chunk_length != expanded_length:
        logger.warning(
            f"MISMATCH: preprocessed chunks are length {actual_chunk_length}, "
            f"but the collator expects expanded_length={expanded_length}. "
            f"preprocess_wura.py needs to chunk to expanded_length, not max_seq_length, "
            f"for CPT. Truncating/padding this test batch to {expanded_length} just to "
            f"exercise the collator - this is NOT a fix, preprocess_wura.py must be updated."
        )

    examples = []
    for i in range(args.batch_size):
        ids = dataset[i]["input_ids"]
        if len(ids) > expanded_length:
            ids = ids[:expanded_length]
        elif len(ids) < expanded_length:
            ids = ids + [tokenizer.pad_token_id] * (expanded_length - len(ids))
        examples.append({"input_ids": ids})

    batch = collator(examples)

    logger.info("--- Shape checks ---")
    logger.info(f"input_ids:         {tuple(batch['input_ids'].shape)} (expected ({args.batch_size}, {config.max_seq_length}))")
    logger.info(f"labels:            {tuple(batch['labels'].shape)} (expected ({args.batch_size}, {target_length}))")
    logger.info(f"decoder_input_ids: {tuple(batch['decoder_input_ids'].shape)}")
    logger.info(f"attention_mask:    {tuple(batch['attention_mask'].shape)}")

    assert batch["input_ids"].shape == (args.batch_size, config.max_seq_length), "input_ids shape mismatch"
    assert batch["labels"].shape == (args.batch_size, target_length), "labels shape mismatch"
    assert batch["decoder_input_ids"].shape == batch["labels"].shape, "decoder_input_ids shape mismatch"
    assert batch["attention_mask"].shape == batch["input_ids"].shape, "attention_mask shape mismatch"
    logger.info("All shape checks passed.")

    vocab_size = len(tokenizer)
    sentinel_threshold = vocab_size - 100
    for i in range(args.batch_size):
        input_sentinels = set(t for t in batch["input_ids"][i].tolist() if t >= sentinel_threshold)
        label_sentinels = set(t for t in batch["labels"][i].tolist() if t >= sentinel_threshold)
        if input_sentinels != label_sentinels:
            logger.warning(
                f"Example {i}: sentinel mismatch between input {input_sentinels} "
                f"and labels {label_sentinels}"
            )
        else:
            logger.info(f"Example {i}: {len(input_sentinels)} sentinels, consistent between input and labels.")

    logger.info("--- Decoded example 0 ---")
    logger.info(f"Corrupted input: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)[:300]}...")
    logger.info(f"Target (labels): {tokenizer.decode(batch['labels'][0], skip_special_tokens=False)[:300]}...")

    logger.info("Collator test complete.")


if __name__ == "__main__":
    main()
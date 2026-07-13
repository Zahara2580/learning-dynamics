"""
Tokenise and chunk the WURA corpus for continued pretraining (CPT) of
a T5-family model (T5, ByT5, Nguni-ByT5).

Note: this script only tokenises and chunks the corpus into fixed-length
blocks. It does NOT apply span corruption (masking spans and replacing
them with sentinel tokens) - that happens later, on the fly, at training
time via a data collator (e.g. DataCollatorForT5MLM). Keeping corruption
out of preprocessing means the same fixed-length token chunks can be
reused across training runs without needing to be regenerated.

Run once for the train split and once for the validation split (the
latter is needed for continued_pretrain.py's --eval-input):
    uv run python3 -m src.data_processing.preprocess_wura --input /path/to/raw/wura --model-config configs/models/t5.yaml --split train --output /path/to/preprocessed/t5/train
    uv run python3 -m src.data_processing.preprocess_wura --input /path/to/raw/wura --model-config configs/models/t5.yaml --split validation --output /path/to/preprocessed/t5/validation
"""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase, logging as hf_logging

from src.pretraining.collator import compute_input_and_target_lengths
from src.pretraining.config import ModelConfig

# Span-corruption noise density (shared by all models). The mean noise
# span length is per-model (byt5 uses 20 per its paper; t5/nguni-byt5
# use 3.0) and comes from the model config - it must match what
# DataCollatorForT5MLM uses at training time, since together they
# determine what chunk length preprocessing needs to produce.
NOISE_DENSITY = 0.15

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize and chunk the WURA corpus for a given model config."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the raw corpus on disk, as saved by download_corpus.py.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model YAML config (provides tokenizer and max_seq_length).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        help="Language subset being preprocessed. Default: xho (isiXhosa)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the processed dataset to."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split of the raw DatasetDict to preprocess. Default: train.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes for dataset preprocessing. Default: 1."
    )
    return parser.parse_args()


def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    num_proc: int = 1
) -> Dataset:
    """
    Tokenise raw text and chunk into fixed-length blocks.

    Works identically for T5, ByT5, and Nguni-ByT5: AutoTokenizer
    handles the difference between subword tokens (T5) and raw UTF-8
    bytes (ByT5) internally, so this function doesn't need to know or
    care which kind of tokenizer it was given.

    :param dataset: HuggingFace Dataset with "headline" and "content" columns.
    :param tokenizer: Tokenizer matching the model being pretrained.
    :param block_size: Fixed chunk length (expanded_length, pre-corruption).
    :param num_proc: Number of processes used for preprocessing.
    :return: Dataset of fixed-length token id chunks.
    """
    def _tokenize(examples: dict[str, list[str]]):
        """Join headline and content, then tokenize."""
        texts = [
            f"{headline}\n\n{content}"
            for headline, content in zip(examples["headline"], examples["content"])
        ]
        # No special_tokens_mask here - that's only needed for BERT-style
        # random single-token masking, which doesn't apply to span
        # corruption (corruption happens later, at training time).
        return tokenizer(texts, truncation=False)

    def _chunk(examples: dict[str, list[list[int]]]):
        """Concatenate and chunk tokenized sequences into fixed-length blocks."""
        # Concatenate all examples in the batch into one long sequence
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}

        # Drop the remainder so every block is exactly block_size long
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        # Split the concatenated sequence into fixed-length blocks
        result = {k: [] for k in concatenated.keys()}
        for k, v in concatenated.items():
            for i in range(0, total_length, block_size):
                result[k].append(v[i: i + block_size])
        return result

    hf_logging.set_verbosity_error()

    logger.info(f"Tokenizing {len(dataset):,} documents...")
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenising documents"
    )

    hf_logging.set_verbosity_warning()

    logger.info(f"Chunking into blocks of {block_size} tokens...")
    chunked = tokenized.map(
        _chunk,
        batched=True,
        num_proc=num_proc,
        desc="Chunking sequences"
    )
    logger.info(f"Produced {len(chunked):,} chunks.")

    return chunked


def main() -> None:
    """Main entry point for preprocessing the corpus."""
    args = parse_args()

    # Load the model configuration settings
    config = ModelConfig.from_yaml(args.model_config)
    logger.info(f"Loaded config for {config.model_name_or_path} (max_seq_length={config.max_seq_length})")

    # Chunks need to be longer than max_seq_length before corruption,
    # since span corruption replaces multi-token spans with a single
    # sentinel token, shrinking the sequence. expanded_length is the
    # pre-corruption length that compresses down to exactly
    # max_seq_length after the collator applies corruption at training
    # time - see src/pretraining/collator.py for the corruption logic.
    expanded_length, _ = compute_input_and_target_lengths(
        input_length=config.max_seq_length,
        noise_density=NOISE_DENSITY,
        mean_noise_span_length=config.mean_noise_span_length,
    )
    logger.info(
        f"Chunking to expanded_length={expanded_length} (post-corruption target: "
        f"{config.max_seq_length}, mean_noise_span_length={config.mean_noise_span_length})"
    )

    # Initialise the model's tokenizer. AutoTokenizer works identically
    # here whether the underlying model is T5 (subword), ByT5, or
    # Nguni-ByT5 (both byte-level) - no branching needed.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Load the corpus split from disk
    logger.info(f"Loading raw corpus from {args.input} (split={args.split})...")
    corpus = load_from_disk(args.input)[args.split]

    # Preprocess the corpus
    chunked_corpus = tokenize_and_chunk(
        dataset=corpus,
        tokenizer=tokenizer,
        block_size=expanded_length,
        num_proc=args.nproc
    )

    # Save the preprocessed corpus
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    chunked_corpus.save_to_disk(str(output_path))
    logger.info(f"Saved preprocessed dataset to {output_path}")


if __name__ == '__main__':
    main()
"""
Tokenise the MT (FLORES-200) and D2T (T2X) datasets for fine-tuning a
T5-family model (T5, ByT5, Nguni-ByT5).

Both tasks are framed as text-to-text (seq2seq): a text input, tagged
with a T5-style task prefix, mapped to a text target. Actual padding
and label handling at training time is done by transformers' own
DataCollatorForSeq2Seq - this script only tokenises source/target text
into input_ids/labels, it does not pad or batch anything.
"""

import argparse
import logging
import re
from argparse import Namespace
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase, logging as hf_logging

from src.pretraining.config import ModelConfig

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# T5-style task prefixes. These are prepended to the source text so
# the model can tell the two tasks apart, following the convention
# T5/mT5/ByT5 were pretrained and fine-tuned with (e.g. "translate
# English to German: ..." "summarize: ...").
MT_PREFIX = "translate English to Xhosa: "
D2T_PREFIX = "generate isiXhosa description: "

# Marker tokens T2X uses to delimit triple fields in its .data files.
T2X_TRIPLE_PATTERN = re.compile(
    r"__start_entity__\s*(.*?)\s*__end_entity__\s*"
    r"__start_type__\s*(.*?)\s*__end_type__\s*"
    r"__start_value__\s*(.*?)\s*__end_value__"
)


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize MT (FLORES-200) or D2T (T2X) data for a given model config."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["mt", "d2t"],
        help="Which finetuning task to preprocess."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the raw task data on disk, as saved by download_finetune_data.py.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model YAML config (provides tokenizer and max_seq_length).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the processed dataset to."
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of processes for dataset preprocessing. Default: 1."
    )
    return parser.parse_args()


def parse_t2x_triple(line: str) -> str:
    """
    Parse a single T2X .data line into a plain, model-readable string.

    T2X triples are written like:
        __start_entity__ Aarhus Airport __end_entity__ __start_type__
        cityServed __end_type__ __start_value__ Aarhus, Denmark __end_value__
    This extracts (entity, type, value) and reformats them as a short
    natural-language-ish string the model can condition on, e.g.:
        "Aarhus Airport | cityServed | Aarhus, Denmark"

    :param line: A single raw line from a T2X .data file.
    :return: A plain string representation of the triple.
    """
    match = T2X_TRIPLE_PATTERN.search(line)
    if not match:
        raise ValueError(f"Could not parse T2X triple from line: {line!r}")
    entity, relation, value = match.groups()
    return f"{entity} | {relation} | {value}"


def load_t2x_as_dataset(input_dir: str) -> DatasetDict:
    """
    Load T2X's raw .data/.text file pairs into a DatasetDict with
    "source" (parsed triple) and "target" (isiXhosa verbalisation)
    columns, matching the shape used for MT.

    :param input_dir: Directory containing T2X's train/valid/test .data/.text files.
    :return: DatasetDict with train/valid/test splits.
    """
    input_path = Path(input_dir)
    splits = {}

    for split in ["train", "valid", "test"]:
        data_lines = (input_path / f"{split}.data").read_text().splitlines()
        text_lines = (input_path / f"{split}.text").read_text().splitlines()

        if len(data_lines) != len(text_lines):
            raise ValueError(
                f"[{split}] mismatched line counts: "
                f"{len(data_lines)} triples vs {len(text_lines)} verbalisations"
            )

        sources = [parse_t2x_triple(line) for line in data_lines]
        splits[split] = Dataset.from_dict({"source": sources, "target": text_lines})
        logger.info(f"[d2t] loaded {split}: {len(splits[split]):,} examples")

    return DatasetDict(splits)


def seq2seq_preprocessor(tokenizer: PreTrainedTokenizerBase, prefix: str, max_length: int):
    """
    Create a preprocessing function for seq2seq tasks (MT, D2T).

    Tokenises the prefixed source text as the model input, and the
    target text as the labels. Actual padding is deferred to
    DataCollatorForSeq2Seq at training time, so no padding happens here.

    :param tokenizer: Tokenizer for the model being fine-tuned.
    :param prefix: T5-style task prefix prepended to every source example.
    :param max_length: Maximum sequence length for both source and target.
    :return: Preprocessing function for use with dataset.map().
    """
    def tokenize(examples: dict[str, list[str]]):
        """Tokenize prefixed source text and target text."""
        inputs = [prefix + text for text in examples["source"]]

        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

        # text_target tokenizes the target sequence using the
        # tokenizer's target-side settings (relevant for models with
        # different source/target tokenization behaviour).
        labels = tokenizer(text_target=examples["target"], max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return tokenize


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    max_length: int,
    num_proc: int
) -> DatasetDict:
    """
    Tokenize a seq2seq dataset (MT or D2T) for fine-tuning.

    :param dataset: DatasetDict with "source" and "target" columns.
    :param tokenizer: Tokenizer for the model being fine-tuned.
    :param prefix: T5-style task prefix.
    :param max_length: Maximum sequence length.
    :param num_proc: Number of processes used for preprocessing.
    :return: Tokenized DatasetDict with all splits provided.
    """
    preprocessor = seq2seq_preprocessor(tokenizer, prefix, max_length)
    column_names = dataset[list(dataset.keys())[0]].column_names

    logger.info("Tokenising source/target pairs...")
    processed = dataset.map(
        preprocessor,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc,
        desc="Tokenising seq2seq pairs"
    )

    logger.info(f"Preprocessed splits: { {k: len(v) for k, v in processed.items()} }")
    return processed


def main() -> None:
    """Main entry point for preprocessing MT/D2T finetuning data."""
    args = parse_args()

    # Load the model configuration settings
    config = ModelConfig.from_yaml(args.model_config)
    logger.info(f"Model: {config.model_name_or_path} (max_seq_length={config.max_seq_length})")

    # Initialise the model's tokenizer
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    hf_logging.set_verbosity_warning()

    # Load the raw dataset, depending on which task this is
    if args.task == "mt":
        prefix = MT_PREFIX
        logger.info(f"Loading MT dataset from {args.input}...")
        dataset = load_from_disk(args.input)
    else:
        prefix = D2T_PREFIX
        logger.info(f"Loading D2T dataset from {args.input}...")
        dataset = load_t2x_as_dataset(args.input)

    logger.info(f"Loaded splits: { {k: len(v) for k, v in dataset.items()} }")

    # Preprocess the dataset
    processed_dataset = preprocess_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        prefix=prefix,
        max_length=config.max_seq_length,
        num_proc=args.nproc
    )

    # Save the preprocessed dataset to disk
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(output_path))
    logger.info(f"Saved preprocessed dataset to {output_path}")


if __name__ == '__main__':
    main()
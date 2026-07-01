"""
Download the WURA isiXhosa corpus from HuggingFace for CPT.

Dataset: https://huggingface.co/datasets/castorini/wura
"""

import argparse
import logging
import os
from argparse import Namespace
from typing import cast, Optional

from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token

# Configure logging to show timestamps and log level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env 
load_dotenv()

# WURA dataset constants.
# SUPPORTED_LANGUAGES only lists xho for now
DATASET_NAME = "castorini/wura"
SUPPORTED_LANGUAGES = ["xho"]


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the WURA corpus from HuggingFace for CPT."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/corpus",
        help="Directory to save the downloaded dataset to disk."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for downloaded files."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        choices=SUPPORTED_LANGUAGES,
        help="WURA language subset to download. Default: xho (isiXhosa)."
    )
    return parser.parse_args()


def load_wura(language: str, cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load the given WURA language dataset from HuggingFace.

    :param language: WURA language subset to download.
    :param cache_dir: Path to the HuggingFace cache directory.
    :return: Dataset with 'train' and 'validation' splits.
    """
    logger.info(f"Loading {DATASET_NAME} ({language})...")

    # Try to get an HF token (from .env / huggingface-cli login).

    token = get_token()
    if token:
        logger.info("Using HuggingFace authentication")
    else:
        logger.info("No HuggingFace token found, proceeding without authentication")

    # trust_remote_code=True is required because WURA ships with a
    # custom loading script wura.py rather than the newer script-less
    # format. verification_mode="no_checks" avoids a checksum mismatch
    # error specific to loading a single-language config of this dataset.
    dataset = cast(DatasetDict, load_dataset(
        path=DATASET_NAME,
        name=language,
        trust_remote_code=True,
        cache_dir=cache_dir,
        token=token,
        verification_mode="no_checks"
    ))
    return dataset


def log_dataset_info(dataset: DatasetDict) -> None:
    """
    Log basic statistics about the loaded dataset.

    :param dataset: Loaded WURA DatasetDict.
    """
    logger.info(f"Dataset structure: {dataset}")
    logger.info(f"Train samples: {len(dataset['train']):,}")
    logger.info(f"Validation samples: {len(dataset['validation']):,}")


def save_dataset(dataset: DatasetDict, output_dir: str) -> None:
    """
    Save the WURA dataset to disk.

    :param dataset: Loaded WURA DatasetDict to save.
    :param output_dir: Directory path to save the dataset to.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")


def main() -> None:
    """Main entry point for downloading the WURA corpus."""
    args = parse_args()

    # --cache-dir overrides HF_DATASETS_CACHE (set in entrypoint.sh on
    # the HPC) if explicitly passed; otherwise fall back to the env var.
    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")


    output_dir = os.path.join(args.output_dir, args.language)

    dataset = load_wura(language=args.language, cache_dir=cache_dir)
    log_dataset_info(dataset)
    save_dataset(dataset, output_dir)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
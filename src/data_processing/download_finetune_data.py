"""
Download evaluation datasets from HuggingFace and GitHub for downstream
task evaluation (MT and D2T).

Datasets:
    - MT:  https://huggingface.co/datasets/Muennighoff/flores200 (xho_Latn, eng_Latn)
    - D2T: https://github.com/francois-meyer/t2x (raw files, no HF loader)
"""

import argparse
import logging
import os
from argparse import Namespace
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token

# Load environment variables (e.g. HF_TOKEN)
load_dotenv()

# Configure logging to show timestamps and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FLORES-200 language configs. isiXhosa needs the Latin-script suffix,
# and since FLORES is a *parallel* corpus, we also need an English
# config to pair against for MT (per the proposal's MT task).
FLORES_DATASET_NAME = "Muennighoff/flores200"
FLORES_SOURCE_LANG = "eng_Latn"
FLORES_TARGET_LANG = "xho_Latn"
FLORES_SPLITS = ["dev", "devtest"]

# T2X is not on HuggingFace - it's raw text files in a GitHub repo.
# Each split has a paired .data (triples) and .text (isiXhosa
# verbalisations) file, aligned line-by-line.
T2X_BASE_URL = "https://raw.githubusercontent.com/francois-meyer/t2x/main"
T2X_SPLITS = ["train", "valid", "test"]

SUPPORTED_LANGUAGES = ["xho"]


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download MT (FLORES-200) and D2T (T2X) evaluation datasets."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mt", "d2t"],
        choices=["mt", "d2t"],
        help="Evaluation tasks to download. Defaults to both."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="xho",
        choices=SUPPORTED_LANGUAGES,
        help="Language subset to download. Default: xho (isiXhosa)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluation",
        help="Root directory to save evaluation datasets."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for downloaded files."
    )
    return parser.parse_args()


def download_flores(output_dir: str, cache_dir: str | None = None) -> None:
    """
    Download and save the FLORES-200 eng_Latn/xho_Latn parallel dataset
    for MT evaluation.

    :param output_dir: Root directory to save evaluation datasets.
    :param cache_dir: HuggingFace cache directory.
    """
    logger.info(f"Loading FLORES-200 ({FLORES_SOURCE_LANG} / {FLORES_TARGET_LANG})...")

    token = get_token()
    if token:
        logger.info("Using HuggingFace authentication")
    else:
        logger.info("No HuggingFace token found, proceeding without authentication")

    # Each language is its own config in this dataset. Sentence `id`s
    # are aligned across configs, so loading both and joining on `id`
    # gives us parallel eng<->xho sentence pairs for MT.
    source = load_dataset(FLORES_DATASET_NAME, FLORES_SOURCE_LANG, cache_dir=cache_dir, token=token)
    target = load_dataset(FLORES_DATASET_NAME, FLORES_TARGET_LANG, cache_dir=cache_dir, token=token)

    task_dir = Path(output_dir) / "mt"
    task_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in FLORES_SPLITS:
        src_col = f"sentence_{FLORES_SOURCE_LANG}"
        tgt_col = f"sentence_{FLORES_TARGET_LANG}"
        splits[split] = Dataset.from_dict({
            "id": source[split]["id"],
            "source": source[split][src_col],
            "target": target[split][tgt_col],
        })
        logger.info(f"[mt] {split}: {len(splits[split]):,} sentence pairs")

    dataset = DatasetDict(splits)
    dataset.save_to_disk(str(task_dir))
    logger.info(f"[mt] saved to {task_dir}")


def _download_file(url: str, dest: Path) -> None:
    """
    Download a single raw file over HTTP.

    :param url: URL of the file to download.
    :param dest: Local path to save the file to.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def download_t2x(output_dir: str) -> None:
    """
    Download T2X's raw .data/.text files for D2T evaluation.

    T2X isn't distributed through HuggingFace, so files are fetched
    directly from GitHub rather than via load_dataset().

    :param output_dir: Root directory to save evaluation datasets.
    """
    logger.info("Downloading T2X (data-to-text)...")

    task_dir = Path(output_dir) / "d2t"
    task_dir.mkdir(parents=True, exist_ok=True)

    for split in T2X_SPLITS:
        for extension in ["data", "text"]:
            filename = f"{split}.{extension}"
            url = f"{T2X_BASE_URL}/{filename}"
            dest = task_dir / filename
            logger.info(f"[d2t] downloading {filename}...")
            _download_file(url, dest)

        # Log a quick line count as a sanity check that both files
        # in the pair are aligned (same number of lines).
        data_lines = (task_dir / f"{split}.data").read_text().splitlines()
        text_lines = (task_dir / f"{split}.text").read_text().splitlines()
        logger.info(f"[d2t] {split}: {len(data_lines):,} triples, {len(text_lines):,} verbalisations")

    logger.info(f"[d2t] saved to {task_dir}")


def main() -> None:
    """Main entry point for downloading evaluation datasets."""
    args = parse_args()

    logger.info(f"Downloading tasks: {args.tasks}")
    logger.info(f"Language: {args.language}")

    if "mt" in args.tasks:
        download_flores(output_dir=args.output_dir, cache_dir=args.cache_dir)

    if "d2t" in args.tasks:
        download_t2x(output_dir=args.output_dir)

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
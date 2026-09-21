"""Download English–isiXhosa FLORES validation and test pairs."""

import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token

logger = logging.getLogger(__name__)
FLORES_DATASET_NAME = "Muennighoff/flores200"
FLORES_SOURCE_LANG = "eng_Latn"
FLORES_TARGET_LANG = "xho_Latn"
FLORES_SPLITS = ["dev", "devtest"]


def download_flores(output_dir: str, cache_dir: str | None = None) -> None:
    """Download and save the FLORES-200 eng_Latn/xho_Latn parallel dataset for MT evaluation."""
    logger.info(f'Loading FLORES-200 ({FLORES_SOURCE_LANG} / {FLORES_TARGET_LANG})')
    token = get_token()
    if token:
        logger.info('Using HuggingFace authentication')
    else:
        logger.info('No HuggingFace token found, proceeding without authentication')
    source = load_dataset(FLORES_DATASET_NAME, FLORES_SOURCE_LANG, cache_dir=cache_dir, token=token)
    target = load_dataset(FLORES_DATASET_NAME, FLORES_TARGET_LANG, cache_dir=cache_dir, token=token)
    task_dir = Path(output_dir) / 'mt'
    task_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in FLORES_SPLITS:
        splits[split] = Dataset.from_dict({
            'id': source[split]['id'],
            'source': source[split]['sentence'],
            'target': target[split]['sentence'],
        })
        logger.info(f'[mt] {split}: {len(splits[split]):} sentence pairs')
    dataset = DatasetDict(splits)
    dataset.save_to_disk(str(task_dir))
    logger.info(f'[mt] saved to {task_dir}')


def main() -> None:
    """Download the dataset to the fine-tuning data directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download FLORES data.")
    parser.add_argument("--output-dir", default="data/finetune", help="Fine-tuning data root.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    args = parser.parse_args()
    download_flores(args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()

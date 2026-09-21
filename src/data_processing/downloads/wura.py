"""Download and save WURA corpora for continued pretraining."""

import argparse
import logging
import os
from argparse import Namespace
from typing import cast, Optional

from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
DATASET_NAME = 'castorini/wura'
SUPPORTED_LANGUAGES = ['xho', 'eng']


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Download the WURA corpus from HuggingFace for CPT.')
    parser.add_argument('--output-dir', type=str, default='data/corpus', help='Directory to save the downloaded dataset to disk.')
    parser.add_argument('--cache-dir', type=str, default=None, help='HuggingFace cache directory for downloaded files.')
    parser.add_argument('--language', type=str, default='xho', choices=SUPPORTED_LANGUAGES, help='WURA language subset to download. Default: xho (isiXhosa).')
    parser.add_argument('--level', type=str, default='passage', choices=['document', 'passage'], help='Corpus granularity; defaults to passages.')
    return parser.parse_args()


def load_wura(language: str, cache_dir: Optional[str] = None, level: str = 'document') -> DatasetDict:
    """Load the given WURA language dataset from HuggingFace."""
    logger.info(f'Loading {DATASET_NAME} ({language}, level={level})...')
    token = get_token()
    if token:
        logger.info('Using HuggingFace authentication')
    else:
        logger.info('No HuggingFace token found, proceeding without authentication')
    dataset = cast(DatasetDict, load_dataset(
        path=DATASET_NAME,
        name=language,
        level=level,
        trust_remote_code=True,
        cache_dir=cache_dir,
        token=token,
        verification_mode='no_checks',
    ))
    return dataset


def log_dataset_info(dataset: DatasetDict) -> None:
    """Log basic statistics about the loaded dataset."""
    logger.info(f'Dataset structure: {dataset}')
    logger.info(f"Train samples: {len(dataset['train']):}")
    logger.info(f"Validation samples: {len(dataset['validation']):}")


def save_dataset(dataset: DatasetDict, output_dir: str) -> None:
    """Save the WURA dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info(f'Dataset saved to {output_dir}')


def main() -> None:
    """Main entry point for downloading the WURA corpus."""
    args = parse_args()
    cache_dir = args.cache_dir or os.environ.get('HF_DATASETS_CACHE')
    suffix = args.language if args.level == 'document' else f'{args.language}-passage'
    output_dir = os.path.join(args.output_dir, suffix)
    dataset = load_wura(language=args.language, cache_dir=cache_dir, level=args.level)
    log_dataset_info(dataset)
    save_dataset(dataset, output_dir)
    logger.info('Download complete.')


if __name__ == '__main__':
    main()

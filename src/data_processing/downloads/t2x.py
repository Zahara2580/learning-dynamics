"""Download all T2X training, validation and test files."""

import argparse
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
T2X_BASE_URL = "https://raw.githubusercontent.com/francois-meyer/t2x/main"
T2X_SPLITS = ["train", "valid", "test"]


def _download_file(url: str, dest: Path) -> None:
    """Download a single raw file over HTTP."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)

def download_t2x(output_dir: str) -> None:
    """Download paired triples and references for every T2X split."""
    logger.info('Downloading T2X (data-to-text)...')
    task_dir = Path(output_dir) / 'd2t'
    task_dir.mkdir(parents=True, exist_ok=True)
    for split in T2X_SPLITS:
        for extension in ['data', 'text']:
            filename = f'{split}.{extension}'
            url = f'{T2X_BASE_URL}/{filename}'
            dest = task_dir / filename
            logger.info(f'[d2t] downloading {filename}...')
            _download_file(url, dest)
        data_lines = (task_dir / f'{split}.data').read_text().splitlines()
        text_lines = (task_dir / f'{split}.text').read_text().splitlines()
        logger.info(f'[d2t] {split}: {len(data_lines):} triples, {len(text_lines):} verbalisations')
    logger.info(f'[d2t] saved to {task_dir}')

def main() -> None:
    """Download the dataset to the fine-tuning data directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download T2X data.")
    parser.add_argument("--output-dir", default="data/finetune", help="Fine-tuning data root.")
    args = parser.parse_args()
    download_t2x(args.output_dir)


if __name__ == "__main__":
    main()

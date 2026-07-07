"""
Download a model and measure its actual on-disk size, in both default
(fp32) and half (fp16) precision. Used to plan checkpoint storage
before running real CPT jobs - given 19 checkpoints per model across
3 models, even a few GB of difference per checkpoint adds up fast.

Usage:
    uv run python3 -m src.unit_tests.test_model_size --model-config configs/models/t5.yaml
"""

import argparse
import logging
import shutil
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM

from src.pretraining.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Measure a model's on-disk checkpoint size.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/model_size_test",
        help="Scratch directory to save test checkpoints to (deleted after measuring).",
    )
    return parser.parse_args()


def get_dir_size_gb(path: Path) -> float:
    """
    Recursively sum the size of all files under path, in GB.

    :param path: Directory to measure.
    :return: Total size in gigabytes.
    """
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 ** 3)


def main() -> None:
    args = parse_args()
    config = ModelConfig.from_yaml(args.model_config)
    output_dir = Path(args.output_dir)

    logger.info(f"Loading {config.model_name_or_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameter count: {num_params:,} ({num_params / 1e9:.2f}B)")

    # --- Measure fp32 (default) size ---
    fp32_dir = output_dir / "fp32"
    logger.info(f"Saving fp32 checkpoint to {fp32_dir}...")
    model.save_pretrained(fp32_dir)
    fp32_size = get_dir_size_gb(fp32_dir)
    logger.info(f"fp32 checkpoint size: {fp32_size:.2f} GB")

    # --- Measure fp16 (half precision) size ---
    fp16_dir = output_dir / "fp16"
    logger.info(f"Saving fp16 checkpoint to {fp16_dir}...")
    model_fp16 = model.half()
    model_fp16.save_pretrained(fp16_dir)
    fp16_size = get_dir_size_gb(fp16_dir)
    logger.info(f"fp16 checkpoint size: {fp16_size:.2f} GB")

    # --- Project total storage across the full checkpoint schedule ---
    num_checkpoints = 19  # per proposal's two-phase schedule
    logger.info("--- Storage projection (this model only) ---")
    logger.info(f"{num_checkpoints} checkpoints x fp32 ({fp32_size:.2f} GB) = {num_checkpoints * fp32_size:.1f} GB")
    logger.info(f"{num_checkpoints} checkpoints x fp16 ({fp16_size:.2f} GB) = {num_checkpoints * fp16_size:.1f} GB")

    # --- Clean up test files ---
    logger.info(f"Cleaning up test directory {output_dir}...")
    shutil.rmtree(output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
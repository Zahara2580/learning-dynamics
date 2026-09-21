"""Locate the latest checkpoint for resuming training."""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the checkpoint directory with the highest step or None"""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"No output directory: {output_dir}")
        return None

    checkpoints = []
    for entry in output_path.iterdir():
        if entry.is_dir():
            match = CHECKPOINT_DIR_PATTERN.match(entry.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, entry))

    if not checkpoints:
        logger.info(f"No checkpoints in {output_dir}")
        return None

    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]

    logger.info(f"Resuming checkpoint: {latest_path} step {latest_step}")
    return str(latest_path)

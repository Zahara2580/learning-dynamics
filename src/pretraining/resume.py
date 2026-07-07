"""
Utility for locating the most recent checkpoint in a training output
directory, so an interrupted CPT run can be resumed.

Hugging Face's Trainer already handles the actual "resume from step N"
logic internally (it saves optimizer state, scheduler state, and step
count alongside model weights, and correctly skips already-seen data
when resumed) - this utility only needs to find *which* checkpoint
folder to point Trainer at.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Trainer saves checkpoints as subfolders named "checkpoint-<step>"
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint folder in a training output directory.

    :param output_dir: Directory where Trainer saves checkpoint-<step> subfolders.
    :return: Path to the checkpoint with the highest step number, or None if
        no checkpoints exist (e.g. first run, nothing to resume).
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"{output_dir} does not exist yet - nothing to resume, starting fresh.")
        return None

    checkpoints = []
    for entry in output_path.iterdir():
        if entry.is_dir():
            match = CHECKPOINT_DIR_PATTERN.match(entry.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, entry))

    if not checkpoints:
        logger.info(f"No checkpoint-<step> folders found in {output_dir} - starting fresh.")
        return None

    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]

    logger.info(
        f"Found {len(checkpoints)} checkpoint(s) in {output_dir}. "
        f"Resuming from the latest: {latest_path} (step {latest_step})"
    )
    return str(latest_path)
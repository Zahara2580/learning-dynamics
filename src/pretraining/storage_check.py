"""
Storage-checking utility for the UCT HPC's `myquota` command (an alias
for the actual /scratch/slurm/bin/purequota executable - subprocess
calls bypass shell aliases entirely, so the real path is used directly).

Used before saving a checkpoint during CPT, to fail cleanly (raise an
error, let the job stop) rather than crash mid-write if there isn't
enough scratch space left - a half-written checkpoint is worse than no
checkpoint at all, since it's both corrupted and still uses disk space.
"""

import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# Safety margin: refuse to save if less than this many GB would remain
# after saving, even if the checkpoint would technically just fit.
SAFETY_MARGIN_GB = 2.0

# Real path to the quota-checking executable. `myquota` is a shell
# alias for this, and aliases aren't visible to subprocess calls.
PUREQUOTA_PATH = "/scratch/slurm/bin/purequota"


def get_scratch_available_gb() -> float:
    """
    Query the HPC's quota tool and parse out available scratch space.

    :return: Available scratch space in GB.
    :raises RuntimeError: If the quota tool cannot be run or its output cannot be parsed.
    """
    try:
        result = subprocess.run(
            [PUREQUOTA_PATH], capture_output=True, text=True, timeout=30, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Could not run {PUREQUOTA_PATH} to check scratch space: {e}")

    # Expected line format: "/scratch         50GB      29GB   58.0%"
    match = re.search(r"/scratch\s+(\d+(?:\.\d+)?)GB\s+(\d+(?:\.\d+)?)GB", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse myquota output:\n{result.stdout}")

    quota_gb = float(match.group(1))
    used_gb = float(match.group(2))
    available_gb = quota_gb - used_gb

    logger.info(f"Scratch quota: {quota_gb:.1f} GB, used: {used_gb:.1f} GB, available: {available_gb:.1f} GB")
    return available_gb


def has_enough_space_for_checkpoint(checkpoint_size_gb: float) -> bool:
    """
    Check whether there is enough scratch space to safely save a
    checkpoint of the given size, including a safety margin.

    :param checkpoint_size_gb: Expected size of the checkpoint to be saved, in GB.
    :return: True if there is enough space, False otherwise.
    """
    available_gb = get_scratch_available_gb()
    required_gb = checkpoint_size_gb + SAFETY_MARGIN_GB

    if available_gb < required_gb:
        logger.warning(
            f"Insufficient scratch space: {available_gb:.1f} GB available, "
            f"need {required_gb:.1f} GB ({checkpoint_size_gb:.1f} GB checkpoint "
            f"+ {SAFETY_MARGIN_GB:.1f} GB safety margin)."
        )
        return False

    logger.info(f"Sufficient scratch space: {available_gb:.1f} GB available, {required_gb:.1f} GB required.")
    return True
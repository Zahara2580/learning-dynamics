"""
Cleanup utility for scratch data - deletes specific categories of
generated/downloaded data on request, to free up quota space.

Everything this script can delete is regenerable (corpus/data can be
re-downloaded, preprocessed chunks can be rebuilt, checkpoints are test
artifacts or resumable via find_latest_checkpoint) - nothing here is
a unique, irreplaceable source of truth. Still defaults to a dry run
so nothing is ever deleted without an explicit --yes.

Usage:
    uv run python3 -m src.unit_tests.cleanup --t5 --dry-run
    uv run python3 -m src.unit_tests.cleanup --t5 --yes
    uv run python3 -m src.unit_tests.cleanup --byt5 --nguni-byt5 --yes
    uv run python3 -m src.unit_tests.cleanup --corpus --finetune --yes
    uv run python3 -m src.unit_tests.cleanup --all --yes
"""

import argparse
import logging
import shutil
from argparse import Namespace
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRATCH_ROOT = Path("/scratch/rmdrak003")

# Maps a flag name to the list of paths it deletes.
TARGETS = {
    "t5": [
        SCRATCH_ROOT / "data" / "preprocessed" / "t5",
        SCRATCH_ROOT / "results" / "t5",
        SCRATCH_ROOT / "results" / "t5-cputest",
    ],
    "byt5": [
        SCRATCH_ROOT / "data" / "preprocessed" / "byt5",
        SCRATCH_ROOT / "results" / "byt5",
    ],
    "nguni-byt5": [
        SCRATCH_ROOT / "data" / "preprocessed" / "nguni-byt5",
        SCRATCH_ROOT / "results" / "nguni-byt5",
    ],
    "corpus": [
        SCRATCH_ROOT / "data" / "corpus",
    ],
    "finetune": [
        SCRATCH_ROOT / "data" / "finetune",
        SCRATCH_ROOT / "data" / "preprocessed" / "mt",
        SCRATCH_ROOT / "data" / "preprocessed" / "d2t",
    ],
    "hf-cache": [
        SCRATCH_ROOT / "hf" / "hub",
    ],
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Delete specific categories of regenerable scratch data.")
    for name in TARGETS:
        parser.add_argument(f"--{name}", action="store_true", help=f"Delete {name} data.")
    parser.add_argument("--all", action="store_true", help="Delete everything (all categories above).")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without this flag, only prints what WOULD be deleted (dry run).",
    )
    return parser.parse_args()


def get_dir_size_gb(path: Path) -> float:
    """Recursively sum the size of all files under path, in GB."""
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 ** 3)


def main() -> None:
    args = parse_args()

    selected = [name for name in TARGETS if args.all or getattr(args, name.replace("-", "_"))]

    if not selected:
        logger.info("No categories selected. Pass --t5, --byt5, --nguni-byt5, --corpus, --finetune, --hf-cache, or --all.")
        return

    logger.info(f"Selected categories: {selected}")

    total_freed_gb = 0.0
    for name in selected:
        for path in TARGETS[name]:
            if not path.exists():
                logger.info(f"[{name}] {path} does not exist, skipping.")
                continue

            size_gb = get_dir_size_gb(path)
            total_freed_gb += size_gb

            if args.yes:
                shutil.rmtree(path)
                logger.info(f"[{name}] Deleted {path} ({size_gb:.2f} GB freed)")
            else:
                logger.info(f"[{name}] Would delete {path} ({size_gb:.2f} GB) - dry run, pass --yes to actually delete")

    action = "Freed" if args.yes else "Would free"
    logger.info(f"{action} approximately {total_freed_gb:.2f} GB total.")

    if not args.yes:
        logger.info("This was a dry run. Rerun with --yes to actually delete.")


if __name__ == "__main__":
    main()
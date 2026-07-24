"""
Back up CPT winner checkpoints to private HF repos (offsite copy).

One private repo per model, each holding all 19 checkpoint-* folders.
Uses upload_large_folder: resumable, so a job that times out mid-upload
continues on resubmission. Run on a WORKER node, never the head node.

Usage:
    uv run python3 -m src.utils.backup_checkpoints --dry-run
    uv run python3 -m src.utils.backup_checkpoints
    uv run python3 -m src.utils.backup_checkpoints --models t5
"""

import argparse
import logging
import os
from argparse import Namespace
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Winner arm per model -> (checkpoints dir, private repo basename).
WINNERS = {
    "t5": ("/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints", "cpt-xhosa-t5-large"),
    "byt5": ("/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints", "cpt-xhosa-byt5-large"),
    "nguni-byt5": ("/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints",
                   "cpt-xhosa-nguni-byt5-large"),
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Back up CPT checkpoints to private HF repos.")
    parser.add_argument("--models", nargs="+", default=list(WINNERS), choices=list(WINNERS))
    parser.add_argument("--dry-run", action="store_true", help="Report what would upload, upload nothing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    logger.info(f"authenticated as {user}")

    for model in args.models:
        path, repo_name = WINNERS[model]
        folder = Path(path)
        if not folder.is_dir():
            logger.warning(f"skip {model}: {folder} not found")
            continue

        repo_id = f"{user}/{repo_name}"
        n_ckpt = len(list(folder.glob("checkpoint-*")))
        logger.info(f"{model}: {n_ckpt} checkpoints in {folder} -> {repo_id} (private)")
        if args.dry_run:
            continue

        api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model")
        logger.info(f"{model}: upload complete")


if __name__ == "__main__":
    main()

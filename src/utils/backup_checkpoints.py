"""
Back up CPT checkpoints to HF repos (offsite copy).

Two modes:

  default    the archived checkpoints/ dir - the 19 weights-only snapshots
             the finetuning sweep consumes. Snapshots what is in the folder
             now (new checkpoints a running trainer adds later are not
             chased), skips any dir still being written, and drops
             optimizer/scheduler/rng (weights-only). Uploads to {repo}.

  --resume   the resume/ dir - the single latest RESUMABLE checkpoint WITH
             optimizer state, for continuing CPT from where it stopped. The
             trainer rewrites this every save interval, so it is copied to a
             local staging dir as a consistent snapshot (verified unchanged
             during the copy) and the frozen copy is uploaded. To {repo}-resume.

Run on a worker node inside screen/sintx (a bare login-shell nohup dies with
the allocation, and heavy hashing/upload must stay off the head node).

Usage:
    uv run python3 -m src.utils.backup_checkpoints --resume --models nguni-byt5 --dry-run
    uv run python3 -m src.utils.backup_checkpoints --resume --models nguni-byt5   # optimizer state, private
    uv run python3 -m src.utils.backup_checkpoints --models nguni-byt5            # weights-only, private
    uv run python3 -m src.utils.backup_checkpoints --models t5 --public
"""

import argparse
import logging
import os
import re
import shutil
import time
from argparse import Namespace
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Arm -> (checkpoints dir, HF repo basename).
#
# Phase 1 (monolingual isiXhosa) and phase 2 (bilingual isiXhosa+English) live
# in separate run-subdirs and go to separate repos, so the two arms can never
# be confused for each other on the Hub.
WINNERS = {
    # phase 1: monolingual isiXhosa
    "t5": ("/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints", "cpt-xhosa-t5-large"),
    "byt5": ("/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints", "cpt-xhosa-byt5-large"),
    "nguni-byt5": ("/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints",
                   "cpt-xhosa-nguni-byt5-large"),
    # phase 2: bilingual isiXhosa + English
    "t5-bilingual": ("/scratch/rmdrak003/results/t5/lafand-bilingual/checkpoints",
                     "cpt-bilingual-t5-large"),
    "byt5-bilingual": ("/scratch/rmdrak003/results/byt5/lafand-bilingual/checkpoints",
                       "cpt-bilingual-byt5-large"),
    "nguni-byt5-bilingual": ("/scratch/rmdrak003/results/nguni-byt5/lafand-bilingual/checkpoints",
                             "cpt-bilingual-nguni-byt5-large"),
}

# Trainer state the finetuning sweep does not need (default mode only).
WEIGHTS_ONLY_IGNORE = ["*/optimizer.pt", "*/scheduler.pt", "*/rng_state*"]
CKPT_RE = re.compile(r"checkpoint-(\d+)")


def checkpoint_step(path: Path) -> int:
    return int(CKPT_RE.fullmatch(path.name).group(1))


def dir_signature(d: Path) -> dict:
    """Map relative-path -> (size, mtime) for every file, to detect changes."""
    return {str(f.relative_to(d)): (f.stat().st_size, f.stat().st_mtime)
            for f in d.rglob("*") if f.is_file()}


# ---------------------------------------------------------------------------
# default mode: archived weights-only checkpoints
# ---------------------------------------------------------------------------

def snapshot_checkpoints(folder: Path, fresh_seconds: int) -> tuple[list[Path], list[str]]:
    """
    List complete checkpoint dirs now; hold back any still being written.

    :return: (dirs sorted by step, names of skipped fresh dirs).
    """
    now = time.time()
    ready, fresh = [], []
    for path in folder.glob("checkpoint-*"):
        if not (path.is_dir() and CKPT_RE.fullmatch(path.name)):
            continue
        newest = max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()),
                     default=path.stat().st_mtime)
        if now - newest < fresh_seconds:
            fresh.append(path.name)
        else:
            ready.append(path)
    ready.sort(key=checkpoint_step)
    fresh.sort(key=lambda n: int(CKPT_RE.fullmatch(n).group(1)))
    return ready, fresh


def upload_size(ckpt_dirs: list[Path], keep_optimizer: bool) -> tuple[int, int]:
    """Total bytes and file count that would upload, after the weights-only filter."""
    skip = () if keep_optimizer else ("optimizer.pt", "scheduler.pt", "rng_state")
    total, n_files = 0, 0
    for d in ckpt_dirs:
        for f in d.rglob("*"):
            if f.is_file() and not any(s in f.name for s in skip):
                total += f.stat().st_size
                n_files += 1
    return total, n_files


# ---------------------------------------------------------------------------
# --resume mode: the live optimizer-state checkpoint
# ---------------------------------------------------------------------------

def newest_resume_checkpoint(resume_folder: Path) -> Path | None:
    """The highest-step checkpoint-* in the resume dir (the current one)."""
    ckpts = [p for p in resume_folder.glob("checkpoint-*")
             if p.is_dir() and CKPT_RE.fullmatch(p.name)]
    return max(ckpts, key=checkpoint_step) if ckpts else None


def stage_resume_checkpoint(
    resume_folder: Path, staging_root: Path,
    fresh_seconds: int = 180, wait: int = 30, retries: int = 6,
) -> Path:
    """
    Copy the newest resumable checkpoint to staging as a consistent snapshot.

    The trainer rewrites resume/ every save interval, so a direct upload can
    catch it mid-save. This waits until the newest checkpoint stops changing,
    copies it locally (fast), then re-checks it did not change or get pruned
    during the copy - retrying with the new newest if it did.

    :return: Path to the staged checkpoint dir.
    """
    staging_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        ckpt = newest_resume_checkpoint(resume_folder)
        if ckpt is None:
            raise RuntimeError(f"no checkpoint-* found in {resume_folder}")

        # Wait out any in-progress save (newest file touched very recently).
        while True:
            newest = max((m for _, m in dir_signature(ckpt).values()), default=0.0)
            if time.time() - newest >= fresh_seconds:
                break
            logger.info(f"  {ckpt.name} still being written; waiting {wait}s")
            time.sleep(wait)

        before = dir_signature(ckpt)
        size_gb = sum(s for s, _ in before.values()) / 1e9
        dest = staging_root / ckpt.name
        if dest.exists():
            shutil.rmtree(dest)
        logger.info(f"  staging {ckpt.name} ({size_gb:.1f} GB with optimizer) -> {dest}")
        shutil.copytree(ckpt, dest)

        # Valid iff our source dir still exists and is byte-for-byte unchanged.
        if ckpt.exists() and dir_signature(ckpt) == before:
            return dest
        logger.warning(f"  {ckpt.name} changed during copy (a save fired); retry {attempt}/{retries}")
        shutil.rmtree(dest, ignore_errors=True)

    raise RuntimeError(
        "resume checkpoint kept changing across retries; "
        "run this when the trainer is idle or between saves")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Back up CPT checkpoints to HF repos.")
    parser.add_argument("--models", nargs="+", default=list(WINNERS), choices=list(WINNERS))
    parser.add_argument("--resume", action="store_true",
                        help="Back up the resume/ optimizer state (to {repo}-resume) instead of "
                             "the archived weights-only checkpoints.")
    parser.add_argument("--public", action="store_true",
                        help="Create a public repo (default private; ignored for --resume).")
    parser.add_argument("--keep-optimizer", action="store_true",
                        help="Default mode: include optimizer/scheduler/rng too.")
    parser.add_argument("--fresh-seconds", type=int, default=180,
                        help="Treat a checkpoint touched more recently than this as mid-write.")
    parser.add_argument("--staging-dir", type=str, default="/scratch/rmdrak003/backup_staging",
                        help="Local staging dir for --resume snapshots.")
    parser.add_argument("--dry-run", action="store_true", help="Report size and selection, upload nothing.")
    return parser.parse_args()


def backup_resume(api: HfApi, user: str, model: str, path: str, repo_name: str, args: Namespace) -> None:
    """Back up the live optimizer-state checkpoint for one model."""
    resume_folder = Path(path).parent / "resume"
    ckpt = newest_resume_checkpoint(resume_folder) if resume_folder.is_dir() else None
    if ckpt is None:
        logger.warning(f"skip {model}: no resume checkpoint under {resume_folder}")
        return

    total = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
    repo_id = f"{user}/{repo_name}-resume"
    logger.info(f"{model}: resume state {ckpt.name} ({total / 1e9:.1f} GB, with optimizer) "
                f"-> {repo_id} (private)")
    if args.dry_run:
        return

    staged = stage_resume_checkpoint(resume_folder, Path(args.staging_dir) / model, args.fresh_seconds)
    try:
        api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=str(staged.parent), repo_type="model")
        logger.info(f"{model}: resume upload complete -> https://huggingface.co/{repo_id}")
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)


def backup_checkpoints(api: HfApi, user: str, model: str, path: str, repo_name: str, args: Namespace) -> None:
    """Back up the archived weights-only checkpoints for one model."""
    folder = Path(path)
    ckpts, fresh = snapshot_checkpoints(folder, args.fresh_seconds)
    if not ckpts:
        logger.warning(f"skip {model}: no settled checkpoints (fresh: {fresh})")
        return

    total, n_files = upload_size(ckpts, args.keep_optimizer)
    repo_id = f"{user}/{repo_name}"
    visibility = "PUBLIC" if args.public else "private"
    logger.info(
        f"{model}: {len(ckpts)} checkpoints [{ckpts[0].name}..{ckpts[-1].name}], "
        f"{n_files} files, {total / 1e9:.1f} GB "
        f"{'(weights-only)' if not args.keep_optimizer else '(full)'} -> {repo_id} ({visibility})")
    if fresh:
        logger.info(f"  holding back {len(fresh)} checkpoint(s) still being written: {fresh}")
    if args.dry_run:
        return

    allow = [f"{d.name}/*" for d in ckpts]
    ignore = [] if args.keep_optimizer else WEIGHTS_ONLY_IGNORE
    api.create_repo(repo_id, private=not args.public, repo_type="model", exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id, folder_path=str(folder), repo_type="model",
        allow_patterns=allow, ignore_patterns=ignore)
    logger.info(f"{model}: upload complete -> https://huggingface.co/{repo_id}")


def main() -> None:
    args = parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    logger.info(f"authenticated as {user}")

    for model in args.models:
        path, repo_name = WINNERS[model]
        if not Path(path).parent.is_dir():
            logger.warning(f"skip {model}: {Path(path).parent} not found")
            continue
        if args.resume:
            backup_resume(api, user, model, path, repo_name, args)
        else:
            backup_checkpoints(api, user, model, path, repo_name, args)


if __name__ == "__main__":
    main()

"""
Restore specific CPT checkpoints from their HuggingFace backup.

The phase-1 (monolingual isiXhosa) checkpoints were deleted from /scratch,
so anything that needs them - the seed-repeat experiment among others - has
to pull them back first. Repo names come from src.utils.backup_checkpoints
WINNERS, the same map the upload used, so there is one source of truth.

Downloads into {dest}/checkpoint-{step}/, which is exactly the layout
run_finetune.discover_checkpoints expects, so the result can be handed
straight to --checkpoints-dir. Nothing in run_finetune changes.

Idempotent: a checkpoint that already looks complete is skipped, so
resubmitting a killed job re-downloads only what is missing.

    uv run python3 -m scripts.seed_experiment.fetch_checkpoints \
        --model byt5 --steps 5000 10000 --dest /scratch/rmdrak003/hf_ckpts/byt5
"""

import argparse
import os
import shutil
from argparse import Namespace
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.utils.backup_checkpoints import WINNERS

# a checkpoint dir is only usable if it has a config and real weights
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Restore CPT checkpoints from HF.")
    p.add_argument("--model", required=True, choices=list(WINNERS))
    p.add_argument("--steps", type=int, nargs="+", required=True,
                   help="CPT steps to fetch. Step 0 is the Hub base model and "
                        "is never fetched here.")
    p.add_argument("--dest", required=True, help="Directory to hold checkpoint-*/")
    p.add_argument("--repo-id", default=None,
                   help="Override owner/name. Default: <your HF user>/<backup name>.")
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    p.add_argument("--no-local", action="store_true",
                   help="Ignore any surviving copy under the original CPT output "
                        "path and always pull from HF.")
    return p.parse_args()


def looks_complete(d: Path) -> bool:
    return d.is_dir() and (d / "config.json").exists() and \
        any((d / w).exists() for w in WEIGHT_FILES)


def dir_size_gb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9


def resolve_repo_id(model: str, override: str | None) -> str:
    """Owner comes from the token, so a missing token fails here with advice."""
    if override:
        return override
    _, repo_name = WINNERS[model]
    token = os.environ.get("HF_TOKEN")
    try:
        user = HfApi(token=token).whoami()["name"]
    except Exception as exc:
        raise SystemExit(
            f"cannot resolve your HF username ({type(exc).__name__}).\n"
            f"  HF_TOKEN is {'set' if token else 'NOT set'} in this shell.\n"
            f"  Interactive shells do not read .env - the sbatch scripts do.\n"
            f"    set -a; source /scratch/rmdrak003/learning-dynamics/.env; set +a\n"
            f"  Or skip the lookup entirely:\n"
            f"    --repo-id <owner>/{repo_name}") from exc
    return f"{user}/{repo_name}"


def main() -> None:
    a = parse_args()
    local_root = Path(WINNERS[a.model][0])

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Anything still on scratch is used in place via a symlink: no download,
    # no second copy of ~2.4GB, and deleting the symlink afterwards cannot
    # touch the real checkpoint.
    wanted = [s for s in a.steps if s != 0]
    if not a.no_local:
        still_local = [s for s in wanted if looks_complete(local_root / f"checkpoint-{s}")]
        for step in still_local:
            link = dest / f"checkpoint-{step}"
            if link.is_symlink() or link.exists():
                if a.force:
                    link.unlink() if link.is_symlink() else shutil.rmtree(link)
                else:
                    print(f"  checkpoint-{step}: already in {dest}, skipping")
                    continue
            link.symlink_to(local_root / f"checkpoint-{step}")
            print(f"  checkpoint-{step}: found on scratch, symlinked "
                  f"(no download) -> {local_root / f'checkpoint-{step}'}")
        wanted = [s for s in wanted if s not in still_local]
        if still_local:
            print(f"  {len(still_local)} checkpoint(s) served locally from {local_root}")

    repo_id = resolve_repo_id(a.model, a.repo_id) if wanted else None
    print(f"model {a.model}   repo {repo_id or '(not needed)'}   dest {dest}")

    if 0 in a.steps:
        print("  step 0 is the un-adapted base model, pulled from the Hub at "
              "train time; not fetched here")

    for step in wanted:
        target = dest / f"checkpoint-{step}"
        if looks_complete(target) and not a.force:
            print(f"  checkpoint-{step}: already present ({dir_size_gb(target):.1f} GB), skipping")
            continue
        if target.exists():
            shutil.rmtree(target)          # partial download from a killed job
        print(f"  checkpoint-{step}: downloading...")
        snapshot_download(repo_id=repo_id, allow_patterns=[f"checkpoint-{step}/*"],
                          local_dir=str(dest), repo_type="model")
        if not looks_complete(target):
            raise SystemExit(
                f"FAIL: {target} has no config.json or weights after download. "
                f"Check that checkpoint-{step} exists in {repo_id}.")
        print(f"  checkpoint-{step}: OK ({dir_size_gb(target):.1f} GB)")

    have = sorted(int(p.name.split("-")[1]) for p in dest.glob("checkpoint-*")
                  if looks_complete(p))
    print(f"\n{dest} now holds checkpoints: {have}")
    missing = [s for s in a.steps if s != 0 and s not in have]
    if missing:
        raise SystemExit(f"FAIL: still missing {missing}")


if __name__ == "__main__":
    main()

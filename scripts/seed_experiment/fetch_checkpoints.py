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
    return p.parse_args()


def looks_complete(d: Path) -> bool:
    return d.is_dir() and (d / "config.json").exists() and \
        any((d / w).exists() for w in WEIGHT_FILES)


def dir_size_gb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    a = parse_args()
    _, repo_name = WINNERS[a.model]
    repo_id = a.repo_id
    if repo_id is None:
        user = HfApi(token=os.environ.get("HF_TOKEN")).whoami()["name"]
        repo_id = f"{user}/{repo_name}"

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"model {a.model}   repo {repo_id}   dest {dest}")

    if 0 in a.steps:
        print("  step 0 is the un-adapted base model, pulled from the Hub at "
              "train time; not fetched here")

    for step in [s for s in a.steps if s != 0]:
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

"""
Verify the HF checkpoint backup is bit-identical to the local scratch
copies - WITHOUT downloading anything. The Hub stores a sha256 for every
LFS file (safetensors) and a git blob id for every small file; we hash
the local files and compare digests. All-OK means the local copy can be
deleted with the backup as the sole source of truth.

Usage (HPC login node is fine - pure CPU hashing, ~10 min per model):
    uv run python3 -m scripts.diagnostics.verify_hf_backup --model t5
    uv run python3 -m scripts.diagnostics.verify_hf_backup --model all
Needs HF_TOKEN in the env for the private nguni repo (source .env).
"""

import argparse
import fnmatch
import hashlib
from argparse import Namespace
from pathlib import Path

from huggingface_hub import HfApi

# Imported, not duplicated: this list previously drifted from the backup
# script's, so a newly-backed-up arm could not be verified until someone
# noticed and edited a second file.
from src.utils.backup_checkpoints import WINNERS as REPOS

# not uploaded by the backup (weights-only) - skip locally too
NOT_UPLOADED = ["*/optimizer.pt", "*/scheduler.pt", "*/rng_state*"]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Verify HF backup == local checkpoints.")
    parser.add_argument("--model", type=str, nargs="+", default=["all"],
                        choices=[*REPOS, "all"],
                        help="One or more arms, or 'all'.")
    parser.add_argument("--user", type=str, default=None,
                        help="HF namespace; default: whoami() of the active token.")
    parser.add_argument("--local-root", type=str, default=None, help="Override the local dir.")
    parser.add_argument("--repo-id", type=str, default=None, help="Override the full repo id.")
    parser.add_argument("--limit-checkpoints", type=int, default=None, help="Smoke test on first N.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Git's blob id: sha1(b'blob <size>\\0' + content) - what the Hub
    reports for small (non-LFS) files."""
    size = path.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def verify_model(api: HfApi, model: str, args: Namespace) -> bool:
    local_root = Path(args.local_root or REPOS[model][0])
    user = args.user or api.whoami()["name"]
    repo_id = args.repo_id or f"{user}/{REPOS[model][1]}"
    print(f"\n=== {model}: {local_root}  vs  {repo_id}")

    ckpt_dirs = sorted((d for d in local_root.glob("checkpoint-*") if d.is_dir()),
                       key=lambda d: int(d.name.split("-")[1]))
    if args.limit_checkpoints:
        ckpt_dirs = ckpt_dirs[:args.limit_checkpoints]
    if not ckpt_dirs:
        print("  NO LOCAL CHECKPOINTS FOUND - nothing to verify")
        return False

    local_files = []
    for d in ckpt_dirs:
        for f in sorted(d.rglob("*")):
            rel = f"{f.parent.name}/{f.name}" if f.parent == d else str(f.relative_to(local_root))
            if f.is_file() and not any(fnmatch.fnmatch(rel, p) for p in NOT_UPLOADED):
                local_files.append((rel, f))

    repo_paths = set(api.list_repo_files(repo_id))
    wanted = [rel for rel, _ in local_files if rel in repo_paths]
    info = {}
    for start in range(0, len(wanted), 50):
        for entry in api.get_paths_info(repo_id, wanted[start:start + 50]):
            info[entry.path] = entry

    ok = mismatched = missing = 0
    hashed_bytes = 0
    for rel, f in local_files:
        entry = info.get(rel)
        if entry is None:
            print(f"  MISSING IN REPO: {rel}")
            missing += 1
            continue
        lfs = getattr(entry, "lfs", None)
        if lfs is not None:
            remote = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
            local = sha256_file(f)
        else:
            remote = entry.blob_id
            local = git_blob_sha1(f)
        hashed_bytes += f.stat().st_size
        if local == remote:
            ok += 1
        else:
            print(f"  HASH MISMATCH: {rel}")
            mismatched += 1

    verdict = "SAFE TO DELETE LOCAL" if mismatched == 0 and missing == 0 else "DO NOT DELETE"
    print(f"  {len(ckpt_dirs)} checkpoints, {ok} files verified bit-identical, "
          f"{mismatched} mismatched, {missing} missing "
          f"({hashed_bytes / 1e9:.1f} GB hashed)")
    print(f"  VERDICT: {verdict}")
    return mismatched == 0 and missing == 0


def main() -> None:
    args = parse_args()
    api = HfApi()
    models = list(REPOS) if "all" in args.model else list(dict.fromkeys(args.model))
    results = {m: verify_model(api, m, args) for m in models}
    print("\n" + "  ".join(f"{m}: {'OK' if v else 'FAILED'}" for m, v in results.items()))
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

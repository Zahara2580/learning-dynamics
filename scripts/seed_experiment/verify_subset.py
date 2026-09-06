"""
Record and verify the identity of the 1,000-example D2T training subset.

The subset is NOT stored anywhere in the original experiment - it is
regenerated on every run by this line in run_finetune.load_task_data:

    idx = sorted(random.Random(42).sample(range(len(train_inputs)), n))

so it is reproducible only as long as that code path and Python's
Mersenne Twister stay put. This script pins it down: it regenerates the
indices, writes them with a hash, and on later runs FAILS if they have
moved. Run it before launching the sweep, on the machine that will run it.

The selection seed (42) is hardcoded and is deliberately NOT the training
seed - varying --seed changes initialisation and data order but leaves the
1,000 examples untouched, which is what the repeat experiment requires.

Run it as a module from the repo root, and CREATE THE RECORD ON THE
MACHINE THAT WILL TRAIN. random.Random(42).sample is not guaranteed
identical across Python versions, and the original ablation ran under the
HPC's python 3.12; a record written under a different interpreter would
be the wrong authority.

    uv run python3 -m scripts.seed_experiment.verify_subset --write
    uv run python3 -m scripts.seed_experiment.verify_subset
"""

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

from src.finetuning.data_t2x import load_t2x_split

RECORD = Path("seed_experiment/d2t_1000_subset.json")
SELECTION_SEED = 42
N_SUBSET = 1000


def main() -> None:
    p = argparse.ArgumentParser(description="Pin the 1,000-example D2T subset.")
    p.add_argument("--data-dir", default="data/finetune/d2t")
    p.add_argument("--record", default=str(RECORD))
    p.add_argument("--write", action="store_true",
                   help="Create the record if absent. Never overwrites.")
    a = p.parse_args()

    train_inputs, _ = load_t2x_split(a.data_dir, "train")
    n_total = len(train_inputs)

    # identical expression to run_finetune.load_task_data
    idx = sorted(random.Random(SELECTION_SEED).sample(range(n_total), N_SUBSET))
    digest = hashlib.sha256(json.dumps(idx).encode()).hexdigest()

    payload = {
        "selection_seed": SELECTION_SEED,
        "n_total_train": n_total,
        "n_subset": N_SUBSET,
        "sha256": digest,
        "indices": idx,
        "generated_by": "random.Random(42).sample(range(n_total), 1000), then sorted",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    print(f"train split      : {n_total} examples")
    print(f"subset           : {N_SUBSET} indices, sha256 {digest[:16]}")
    print(f"python           : {sys.version.split()[0]}")

    record = Path(a.record)
    if record.exists():
        prev = json.loads(record.read_text())
        same_hash = prev["sha256"] == digest
        same_total = prev["n_total_train"] == n_total
        print(f"recorded         : sha256 {prev['sha256'][:16]} "
              f"(python {prev.get('python')})")
        if prev.get("python") != sys.version.split()[0]:
            print(f"note             : record was written under python "
                  f"{prev.get('python')}, running under {sys.version.split()[0]}; "
                  f"the hash check below is what decides whether that matters")
        if not same_total:
            raise SystemExit(
                f"FAIL: train split is {n_total} examples, record says "
                f"{prev['n_total_train']}. The data changed; the subset is not "
                f"comparable with the original experiment.")
        if not same_hash:
            raise SystemExit(
                "FAIL: regenerated subset does not match the recorded one. "
                "random.Random(42).sample has produced a different draw, most "
                "likely a Python version difference. The original 1,000-example "
                "results are NOT comparable with what this machine would run. "
                "Stop and resolve before launching.")
        print("OK: subset matches the record exactly.")
        return

    if not a.write:
        raise SystemExit(f"no record at {record}; rerun with --write to create it")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2))
    print(f"wrote {record}")


if __name__ == "__main__":
    main()

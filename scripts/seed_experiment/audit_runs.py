"""
Check every seed-experiment row for internal consistency.

The concurrency bug deleted a checkpoint another job was still using. That
failure mode is LOUD - discover_checkpoints raises rather than substituting a
different checkpoint - but "it should have crashed" is not evidence. This
re-derives, from each row's own recorded fields, that the run it claims to be
is the run it actually was.

Per row:
  checkpoint_path      must end in checkpoint-{ckpt_step}, or be a bare Hub id
                       when ckpt_step is 0
  seed                 must match the directory it was written into
  config_hash          must be the one expected for the arm
  n_train_examples     1000 for the limited arm, 3859 for the full arm
  n_test_examples      378, the T2X test split
  duplicates           no repeated (model, step, seed, selection)

    uv run python3 -m scripts.seed_experiment.audit_runs
"""

import argparse
import json
import re
from argparse import Namespace
from collections import Counter
from pathlib import Path

# hashes verified against the published ablation rows
ARMS = {"d2t_1000": ("649b6fe995e0", 1000), "d2t_full": ("04715bd042d2", 3859)}
STEPS = [0, 5000, 10000]
N_TEST = 378


def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description="Audit seed-experiment result rows.")
    p.add_argument("--root", default="seed_experiment")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = Path(a.root)
    problems, n_rows = [], 0

    for arm, (want_hash, want_n_train) in ARMS.items():
        for seed in a.seeds:
            d = root / f"{arm}_seed{seed}"
            path = d / "results.jsonl"
            if not path.exists():
                print(f"{d.name:26} ABSENT")
                continue
            rows = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
            n_rows += len(rows)
            keys = Counter((r["model"], r["ckpt_step"], r["seed"], r["selection"])
                           for r in rows)
            dups = [k for k, c in keys.items() if c > 1]
            steps_done = sorted({r["ckpt_step"] for r in rows})

            for r in rows:
                tag = f"{d.name} {r['model']} step {r['ckpt_step']} {r['selection']}"
                p_, s_ = r["checkpoint_path"], r["ckpt_step"]
                if s_ == 0:
                    if "checkpoint-" in p_:
                        problems.append(f"{tag}: step 0 loaded {p_}")
                elif not re.search(rf"checkpoint-{s_}$", p_):
                    problems.append(f"{tag}: path {p_} does not end in checkpoint-{s_}")
                if r["seed"] != seed:
                    problems.append(f"{tag}: seed {r['seed']} in a seed{seed} directory")
                if r["config_hash"] != want_hash:
                    problems.append(f"{tag}: config_hash {r['config_hash']} != {want_hash}")
                if r.get("n_train_examples") != want_n_train:
                    problems.append(f"{tag}: n_train {r.get('n_train_examples')} != {want_n_train}")
                if r.get("n_test_examples") != N_TEST:
                    problems.append(f"{tag}: n_test {r.get('n_test_examples')} != {N_TEST}")
            if dups:
                problems.append(f"{d.name}: duplicate keys {dups}")

            missing = [s for s in STEPS if s not in steps_done]
            state = "complete" if not missing else f"missing steps {missing}"
            print(f"{d.name:26} {len(rows):2} rows  steps {steps_done}  {state}")

    print(f"\n{n_rows} rows audited")
    if n_rows == 0:
        # an all-clear over nothing is worse than an error
        raise SystemExit("nothing to audit: no results.jsonl found. Run this on "
                         "the HPC, or point --root at a pulled copy.")
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p_ in problems:
            print("  -", p_)
        raise SystemExit(1)
    print("no inconsistencies: every row's checkpoint path, seed, protocol hash, "
          "training size and test size match what its location claims.")


if __name__ == "__main__":
    main()

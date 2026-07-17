"""
Forward-pass loss diagnostic for nguni-byt5 using the lafand-mt (AfriByT5)
corruption it was actually CPT'd with, instead of our T5-style collator.

Replicates lafand's preprocess.py exactly:
  - 15% of positions sampled i.i.d. uniformly (random.sample), NOT
    span-constructed -> masked runs are geometric, mean ~1.18 bytes
  - consecutive selected positions merged into one run, one sentinel each
    (racha_detection + masking)
  - target = complement masking (unmasked runs -> sentinels, masked
    tokens kept), position 0 always unmasked so input/target sentinel
    numbering aligns (lafand enforced this via the while target[0] loop)

Runs the same masked batch through nguni-byt5 under the three candidate
sentinel conventions (identical masks, only sentinel ids differ):
  A: descending from 383  (T5-style; what our collator's base-384 does)
  B: ascending  from 259  (newer-transformers <extra_id_0>=259)
  C: descending from 258  (lafand code as written under transformers 4.10)

Lowest loss = the convention nguni-byt5 was born with, now tested at its
native ~1.18 span length rather than the span-3/span-20 grid we used
before. Set INCLUDE_BYT5=1 to also run byt5-large as an anchor (expect C
to win there per its paper).

CPU-only (no GPU request needed), ~2-5 min per variant. Run in sintx:
    cd /scratch/rmdrak003/learning-dynamics
    uv run python3 scripts/memory_fit_checks/diagnose_nguni_lafand_style.py

Self-test of just the mask construction (no model, runs anywhere):
    uv run python3 scripts/memory_fit_checks/diagnose_nguni_lafand_style.py --selftest
"""

import os
import random
import sys

if os.path.isdir("/scratch/rmdrak003"):
    os.environ.setdefault("HF_HOME", "/scratch/rmdrak003/hf")

NGUNI_DATA = os.environ.get("NGUNI_DATA", "/scratch/rmdrak003/data/preprocessed/nguni-byt5")
BYT5_DATA = os.environ.get("BYT5_DATA", "/scratch/rmdrak003/data/preprocessed/byt5")
BATCH = 4
PERCENT = 0.15
EOS_ID = 1
PAD_ID = 0

VARIANTS = [
    ("A: descending from 383 (T5-style / our base-384)", lambda k: 383 - k),
    ("B: ascending from 259  (<extra_id_0>=259)", lambda k: 259 + k),
    ("C: descending from 258 (lafand as written)", lambda k: 258 - k),
]


def group_runs(indices):
    """lafand's racha_detection: split sorted indices into consecutive runs."""
    runs, run = [], []
    for i, idx in enumerate(indices):
        run.append(idx)
        if i == len(indices) - 1 or indices[i + 1] != idx + 1:
            runs.append(run)
            run = []
    return runs


def lafand_mask(ids, rng):
    """i.i.d. 15% position sampling, position 0 excluded (lafand's
    while-loop guaranteed the target starts with the first sentinel,
    which requires position 0 unmasked). Returns (masked_run_list,
    unmasked_run_list) over positions."""
    n_mask = int(len(ids) * PERCENT)
    masked_idx = sorted(rng.sample(range(1, len(ids)), n_mask))
    masked_set = set(masked_idx)
    unmasked_idx = [i for i in range(len(ids)) if i not in masked_set]
    return group_runs(masked_idx), group_runs(unmasked_idx)


def build_pair(ids, masked_runs, unmasked_runs, sentinel_fn):
    """Build (input_ids, target_ids) exactly as lafand's masking():
    each run replaced by one sentinel (numbered 0,1,2,... separately for
    input and target), other run positions dropped, EOS appended."""
    replace_in = {}   # position -> sentinel id (input side: masked runs)
    drop_in = set()
    for k, run in enumerate(masked_runs):
        replace_in[run[0]] = sentinel_fn(k)
        drop_in.update(run[1:])
    replace_tg = {}   # target side: unmasked runs
    drop_tg = set()
    for k, run in enumerate(unmasked_runs):
        replace_tg[run[0]] = sentinel_fn(k)
        drop_tg.update(run[1:])

    inp = [replace_in.get(i, t) for i, t in enumerate(ids) if i not in drop_in]
    tgt = [replace_tg.get(i, t) for i, t in enumerate(ids) if i not in drop_tg]
    return inp + [EOS_ID], tgt + [EOS_ID]


def selftest():
    rng = random.Random(0)
    ids = [rng.randint(3, 258) for _ in range(568)]
    masked_runs, unmasked_runs = lafand_mask(ids, random.Random(1))
    lens = [len(r) for r in masked_runs]
    print(f"masked positions: {sum(lens)} / {len(ids)} = {sum(lens)/len(ids):.3f}")
    print(f"masked runs: {len(masked_runs)}, mean run length = {sum(lens)/len(lens):.3f} "
          f"(geometric expectation 1/(1-0.15) = {1/0.85:.3f})")
    for name, fn in VARIANTS:
        inp, tgt = build_pair(ids, masked_runs, unmasked_runs, fn)
        sentinels_used = [fn(k) for k in range(len(masked_runs))]
        assert tgt[0] == fn(0), "target must start with the first sentinel"
        assert min(sentinels_used) >= 0 and max(sentinels_used) <= 383, "sentinel out of vocab"
        print(f"{name}: input len {len(inp)}, target len {len(tgt)}, "
              f"sentinels {sentinels_used[0]}..{sentinels_used[-1]}")
    print("selftest OK")


def run_model(model_name, data_path):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSeq2SeqLM

    print(f"\n=== {model_name} (data: {data_path}) ===")
    dataset = load_from_disk(data_path)
    chunks = [dataset[i]["input_ids"] for i in range(BATCH)]
    print(f"chunk length {len(chunks[0])}, batch {BATCH}")

    # one fixed mask per example, shared across all variants
    masks = [lafand_mask(ids, random.Random(100 + i)) for i, ids in enumerate(chunks)]
    lens = [len(r) for m, _ in masks for r in m]
    print(f"mean masked-run length across batch: {sum(lens)/len(lens):.3f} "
          f"(lafand geometric expectation ~1.18); runs per example ~{len(lens)//BATCH}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    for name, fn in VARIANTS:
        inputs, targets = [], []
        for ids, (masked_runs, unmasked_runs) in zip(chunks, masks):
            inp, tgt = build_pair(ids, masked_runs, unmasked_runs, fn)
            inputs.append(inp)
            targets.append(tgt)
        in_len = max(len(x) for x in inputs)
        tg_len = max(len(x) for x in targets)
        input_ids = torch.tensor([x + [PAD_ID] * (in_len - len(x)) for x in inputs])
        attention_mask = (input_ids != PAD_ID).long()
        # -100 padding so padding is excluded from the loss (lafand padded
        # with 0 which pollutes loss; we want a clean per-token mean here)
        labels = torch.tensor([x + [-100] * (tg_len - len(x)) for x in targets])

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print(f"{name}\n  -> loss = {out.loss.item():.4f}")

    del model


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)

    run_model("francois-meyer/nguni-byt5-large", NGUNI_DATA)
    if os.environ.get("INCLUDE_BYT5") == "1":
        run_model("google/byt5-large", BYT5_DATA)

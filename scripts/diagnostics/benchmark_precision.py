"""
How much slower and how much bigger is CPT without bf16 autocast?

Measured on the actual GPU with the actual models, because the answer depends
on the card, the sequence length and the batch size - not on a spec sheet.

Runs real forward/backward/AdamW steps in three configurations:
  fp32 params + bf16 autocast   what phase 1's SETTING would have been if the
                                params had not also been cast (i.e. the fix
                                that keeps autocast)
  fp32 params, no autocast      the proposed configuration
  bf16 params                   what phase 1 ACTUALLY ran (broken, for scale)

    sbatch --wrap "..."   or inside sintx with a GPU:
    uv run python3 scripts/diagnostics/benchmark_precision.py --model byt5
"""

import argparse
import time

import torch
from transformers import AutoModelForSeq2SeqLM

MODELS = {"t5": ("google-t5/t5-large", 8), "byt5": ("google/byt5-large", 4),
          "nguni-byt5": ("francois-meyer/nguni-byt5-large", 4)}
SEQ = 512
WARMUP, MEASURE = 3, 10


def run(model_id, param_dtype, autocast, batch, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=param_dtype).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(4, 200, (batch, SEQ), device=device)
    labels = torch.randint(4, 200, (batch, SEQ), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labels).loss
        else:
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(MEASURE):
        step()
    torch.cuda.synchronize()
    per_step = (time.time() - t0) / MEASURE
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return per_step, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="byt5", choices=list(MODELS))
    ap.add_argument("--batch", type=int, default=None,
                    help="per-device micro-batch; default matches the CPT launch script")
    ap.add_argument("--accum", type=int, default=None,
                    help="gradient accumulation, for the wall-clock projection")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    model_id, default_batch = MODELS[a.model]
    batch = a.batch or default_batch
    accum = a.accum or (1024 // batch)
    print(f"{a.model}  batch {batch} x accum {accum} = effective {batch*accum}  "
          f"seq {SEQ}  on {torch.cuda.get_device_name(0)}\n")

    cfgs = [
        ("fp32 params + bf16 autocast", torch.float32, True),
        ("fp32 params, NO autocast   ", torch.float32, False),
        ("bf16 params (phase 1, broken)", torch.bfloat16, False),
    ]
    print(f"  {'configuration':32} {'s / micro-step':>15} {'peak GB':>9} {'10k steps':>12}")
    base = None
    for label, dt, amp in cfgs:
        try:
            per_step, peak = run(model_id, dt, amp, batch, "cuda")
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:32} {'OOM':>15} {'-':>9} {'-':>12}")
            torch.cuda.empty_cache()
            continue
        hours = per_step * accum * 10_000 / 3600
        base = base or per_step
        print(f"  {label:32} {per_step:>15.4f} {peak:>9.1f} {hours:>10.1f} h"
              f"   ({per_step/base:.2f}x)")

    print(f"\n  '10k steps' projects one full CPT run: per-micro-step x {accum} "
          f"accumulation x 10,000 optimiser steps.")
    print(f"  Real jobs are slower - data loading, evaluation every 200 steps, "
          f"and checkpoint writes are not in this loop.")


if __name__ == "__main__":
    main()

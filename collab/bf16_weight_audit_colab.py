"""
Did CPT actually move the weights? A bf16 precision audit.

Standalone - does not depend on the probing notebook. Paste each
"# %% CELL n" block into its own Colab cell, in order.

WHY THIS EXISTS
    src/pretraining/continued_pretrain_lafand.py:234 loads the model with
    torch_dtype=torch.bfloat16, so AdamW wrote its updates into bf16
    parameters with no fp32 master copy. bf16 keeps 8 mantissa bits, so the
    gap between representable values scales with magnitude:

        |w| ~ 18     -> gap 0.125      an lr=1e-4 update rounds away
        |w| ~ 0.05   -> gap 0.0002     an lr=1e-4 update lands

    T5 embeddings sit near |w| = 18. Every optimiser step on them was
    computed, applied, and discarded by rounding. Transformer-block weights
    are ~400x smaller and did move.

    CELL 4 proves the mechanism with no downloads. CELL 5 measures whether
    it actually happened, by counting weights that are bit-identical to the
    un-adapted base model after 10,000 steps of training.

Everything is compared in bf16 - the dtype the checkpoints are stored in AND
the dtype CPT trained in - so "identical" means the optimiser genuinely never
moved that value, not that a cast hid the difference.
"""

# %% CELL 1 - install
# !pip install -q "transformers>=5.12.1" "huggingface_hub>=0.24" numpy

# %% CELL 2 - login (only needed for the PRIVATE nguni repo)
# from huggingface_hub import notebook_login
# notebook_login()

# %% CELL 3 - imports and configuration
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM

MODELS = {
    "t5":         ("google-t5/t5-large", "ChonkeyJellyfish/cpt-xhosa-t5-large"),
    "byt5":       ("google/byt5-large", "ChonkeyJellyfish/cpt-xhosa-byt5-large"),
    "nguni-byt5": ("francois-meyer/nguni-byt5-large",
                   "ChonkeyJellyfish/cpt-xhosa-nguni-byt5-large"),   # PRIVATE
}

RUN_MODELS = ["t5", "byt5"]        # add "nguni-byt5" after notebook_login()
STEP = 10000                       # final CPT checkpoint
LR = 1e-4                          # the CPT learning rate, from configs/models/*.yaml
CKPT_DIR = Path("/content/_ckpt")

print(f"torch {torch.__version__}")

# %% CELL 4 - PROVE THE MECHANISM (instant, no downloads)
#
# Run 10,000 real AdamW steps at the CPT learning rate on a single parameter,
# once with bf16 params (what CPT did) and once with fp32 params (proper
# mixed precision, what Siya's setup does). Same optimiser, same gradient,
# same number of steps - the only difference is parameter dtype.


def bf16_gap(v):
    """Gap between adjacent representable bf16 values near |v| (8-bit mantissa)."""
    return 2.0 ** (math.floor(math.log2(abs(v))) - 7) if v else 0.0


def run_steps(magnitude, dtype, steps=10000, lr=LR, grad=0.01):
    p = torch.nn.Parameter(torch.full((8,), magnitude, dtype=dtype))
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=0.0)
    start = p.detach().clone().float()
    for _ in range(steps):
        opt.zero_grad()
        p.grad = torch.full_like(p, grad)
        opt.step()
    return (p.detach().float() - start).abs().mean().item()


print(f"10,000 AdamW steps at lr={LR}, identical gradients, only dtype differs\n")
print(f"{'weight |w|':>12} {'bf16 gap':>11} {'bf16 moved':>12} {'fp32 moved':>12}   verdict")
for mag, label in [(18.0, "T5 embeddings"), (1.0, ""), (0.05, "attention/FFN")]:
    mb = run_steps(mag, torch.bfloat16)
    mf = run_steps(mag, torch.float32)
    verdict = "FROZEN - updates discarded" if mb < 1e-6 else f"trains, but {mf/max(mb,1e-9):.0f}x slower than fp32"
    print(f"{mag:>12} {bf16_gap(mag):>11.6f} {mb:>12.6f} {mf:>12.6f}   {verdict}"
          + (f"   <- {label}" if label else ""))

print("\nNothing about this depends on our checkpoints: it is a property of the")
print("number format. CELL 5 checks whether it actually bit our training run.")

# %% CELL 5 - THE AUDIT: how many weights are bit-identical to the base model?


def fetch_checkpoint(model_name, step):
    _, repo_id = MODELS[model_name]
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    snapshot_download(repo_id=repo_id, allow_patterns=[f"checkpoint-{step}/*"],
                      local_dir=str(CKPT_DIR))
    return str(CKPT_DIR / f"checkpoint-{step}")


def bucket(k):
    # lm_head IS shared.weight when tie_word_embeddings=True (t5-large: True),
    # so it belongs with the embeddings, not with attention/ffn.
    if any(t in k for t in ("shared", "embed_tokens", "lm_head")):
        return "embeddings"
    if "layer_norm" in k:
        return "layernorm"
    if "relative_attention_bias" in k:
        return "rel_attn_bias"
    return "attention/ffn"


def load_params_bf16(path):
    """named_parameters() deduplicates tied tensors; state_dict() would count
    the embedding matrix 4x (shared / encoder.embed_tokens /
    decoder.embed_tokens / lm_head). bf16 halves memory AND is exactly what
    CPT saw: it loaded the fp32 base cast down to bf16."""
    m = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.bfloat16)
    out = {k: v.detach().clone() for k, v in m.named_parameters()}
    del m
    return out


for model_name in RUN_MODELS:
    base_id, _ = MODELS[model_name]
    print(f"\n{'=' * 78}\n{model_name}: base vs checkpoint-{STEP}, compared in bf16")
    try:
        base = load_params_bf16(base_id)
        cpt = load_params_bf16(fetch_checkpoint(model_name, STEP))
    except Exception as e:
        print(f"  SKIPPED: {type(e).__name__}: {str(e)[:120]}")
        continue

    # A weight can absorb an lr-sized update only if its own bf16 gap is under
    # ~2*lr. Since gap = 2^(floor(log2|w|) - 7), that means |w| below this:
    MOVABLE_BELOW = 2.0 ** (math.ceil(math.log2(2 * LR * 128)) - 1)

    agg = defaultdict(lambda: [0, 0, 0.0, 0.0, 0])   # n, n_identical, sum w^2, max|d|, n_movable
    for k, x in base.items():
        if k not in cpt or not x.is_floating_point():
            continue
        g = agg[bucket(k)]
        g[0] += x.numel()
        g[1] += (x == cpt[k]).sum().item()
        g[2] += x.double().pow(2).sum().item()
        g[3] = max(g[3], (cpt[k].float() - x.float()).abs().max().item())
        # per-weight, not per-group: real magnitudes span orders of magnitude
        # around the RMS, and the small ones have fine enough gaps to move.
        g[4] += (x.float().abs() < MOVABLE_BELOW).sum().item()

    print(f"\n  a bf16 weight can absorb an lr={LR:g} update only if |w| < {MOVABLE_BELOW:g}")
    print(f"\n{'group':16s} {'params':>13} {'% moved':>9} {'% COULD move':>13} "
          f"{'rms |w|':>9} {'bf16 gap':>10} {'max |delta|':>12}")
    for name, (n, same, sq, mx, movable) in sorted(agg.items()):
        rms = float(np.sqrt(sq / n))
        print(f"{name:16s} {n:>13,} {100 * (n - same) / n:>8.2f}% "
              f"{100 * movable / n:>12.2f}% {rms:>9.4f} "
              f"{bf16_gap(rms):>10.6f} {mx:>12.2e}")
    print("\n  '% moved' should track '% COULD move' - that correspondence IS the")
    print("  mechanism: what changed is what bf16 had the resolution to record.")

    tot_n = sum(v[0] for v in agg.values())
    tot_same = sum(v[1] for v in agg.values())
    print(f"\n  whole model: {100 * tot_same / tot_n:.2f}% of "
          f"{tot_n:,} parameters are bit-identical to the base model")

    del base, cpt
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

# %% CELL 6 - how to read this
print("""
READING THE TABLE

  % NEVER moved   parameters bit-identical to the un-adapted base after
                  10,000 training steps. High here = the optimiser's updates
                  were rounded away rather than applied.

  bf16 gap        spacing between representable bf16 values at that weight
                  magnitude. An AdamW step is at most ~lr, so if the gap is
                  much bigger than 2*lr, no single update can ever change
                  the weight - regardless of the gradient.

  Compare the embeddings row against attention/ffn. Same model, same
  optimiser, same steps: the only difference is weight magnitude, and
  therefore the resolution bf16 had available to record the update.

THE FIX, for any re-run
  continued_pretrain_lafand.py already accepts --model-dtype fp32. Passing it
  keeps parameters in fp32 while bf16=True still autocasts activations - real
  mixed precision. Costs roughly double the optimiser memory, so the batch
  size or gradient accumulation may need adjusting.
""")

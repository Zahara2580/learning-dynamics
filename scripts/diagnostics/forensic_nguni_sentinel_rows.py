"""
Weight forensics: which token IDs did nguni-byt5's continued pretraining
actually train?

nguni-byt5-large was initialized FROM google/byt5-large, so any embedding
row its CPT never used still matches byt5's bit-for-bit (input-embedding
rows only receive gradient when their token appears in a batch; with
weight_decay=0 unused rows never move). Diffing the two models row-by-row
is therefore a direct physical readout of which sentinel convention the
lafand-mt-based CPT actually used:

  - sentinels descending from 383  -> diffs at 383 decaying downward
  - sentinels ascending from 259   -> diffs at 259 decaying upward
  - sentinels descending from 258  -> diffs bleeding down into byte range
                                      (lafand preprocess.py as written,
                                      transformers 4.10)

The input embedding (shared.weight) is the primary evidence. lm_head gets
dense gradients through the softmax for every row at every step, so all
its rows will have drifted - reported only as secondary context.

CPU-only, no GPU. Models load from the HF cache (already downloaded).
Run in a sintx shell:
    cd /scratch/rmdrak003/learning-dynamics
    uv run python3 scripts/memory_fit_checks/forensic_nguni_sentinel_rows.py
"""

import gc
import os

# Keep the HF cache on scratch even in a bare shell (prevents the
# duplicated-cache problem from before). Respect HF_HOME if already set.
if os.path.isdir("/scratch/rmdrak003"):
    os.environ.setdefault("HF_HOME", "/scratch/rmdrak003/hf")

import torch
from transformers import AutoModelForSeq2SeqLM

BYT5 = "google/byt5-large"
NGUNI = "francois-meyer/nguni-byt5-large"


def grab_weights(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach().clone()
    head = model.lm_head.weight.detach().clone()
    del model
    gc.collect()
    return emb, head


print(f"Loading {BYT5}...")
emb_b, head_b = grab_weights(BYT5)
print(f"Loading {NGUNI}...")
emb_n, head_n = grab_weights(NGUNI)
assert emb_b.shape == emb_n.shape, f"shape mismatch: {emb_b.shape} vs {emb_n.shape}"
print(f"embedding shape: {tuple(emb_b.shape)}\n")

emb_diff = (emb_n - emb_b).norm(dim=1)
head_diff = (head_n - head_b).norm(dim=1)
byte_baseline = emb_diff[3:134].mean()  # ordinary ASCII bytes: definitely trained

RANGES = [
    ("specials 0-2 (pad/eos/unk)", 0, 3),
    ("bytes 3-133 (ASCII incl. letters)", 3, 134),
    ("bytes 134-258 (high bytes = paper sentinel zone)", 134, 259),
    ("extras 259-320 (HF extra_id low end)", 259, 321),
    ("extras 321-383 (HF extra_id high end)", 321, 384),
]

print("=== per-range input-embedding diff (L2 per row, nguni vs byt5) ===")
print(f"{'range':<50} {'mean':>10} {'max':>10} {'#exact-zero':>12}")
for name, lo, hi in RANGES:
    d = emb_diff[lo:hi]
    n_zero = int((d == 0).sum())
    print(f"{name:<50} {d.mean():>10.4f} {d.max():>10.4f} {n_zero:>12}")
print(f"\n(reference: trained ASCII byte rows average {byte_baseline:.4f} - "
      "rows well below this were touched rarely or never)\n")

print("=== row-by-row around the 258/259 boundary ===")
for i in range(250, 269):
    print(f"  row {i:>3}: emb diff = {emb_diff[i]:.5f}   head diff = {head_diff[i]:.5f}")

print("\n=== row-by-row at the top of the vocab ===")
for i in range(370, 384):
    print(f"  row {i:>3}: emb diff = {emb_diff[i]:.5f}   head diff = {head_diff[i]:.5f}")

print("\n=== top 25 most-changed rows in 134-383 (sentinel candidate zone) ===")
zone = emb_diff.clone()
zone[:134] = -1
top = torch.topk(zone, 25)
for val, idx in zip(top.values, top.indices):
    print(f"  row {int(idx):>3}: emb diff = {val:.5f}")

# Direction verdict: compare how the diff decays across the extras range.
lo_end = emb_diff[259:290].mean()
hi_end = emb_diff[353:384].mean()
byte_top = emb_diff[200:259].mean()
print("\n=== VERDICT SIGNALS ===")
print(f"extras low end  (259-289) mean diff: {lo_end:.4f}")
print(f"extras high end (353-383) mean diff: {hi_end:.4f}")
print(f"high-byte zone  (200-258) mean diff: {byte_top:.4f}")
print(f"trained-byte baseline (ASCII)      : {byte_baseline:.4f}")
print()
if hi_end > 3 * lo_end and hi_end > 0.1 * byte_baseline:
    print("PATTERN: top-down. Diffs concentrate at 383 and decay downward")
    print("-> CPT sentinels DESCENDED from 383 (T5-style / our collator's base-384 default).")
elif lo_end > 3 * hi_end and lo_end > 0.1 * byte_baseline:
    print("PATTERN: bottom-up. Diffs concentrate at 259 and decay upward")
    print("-> CPT sentinels ASCENDED from 259 (newer-transformers <extra_id_0>=259).")
elif byte_top > byte_baseline and lo_end < 0.1 * byte_baseline and hi_end < 0.1 * byte_baseline:
    print("PATTERN: extras untouched, high bytes hot")
    print("-> CPT sentinels DESCENDED from 258 into the byte range (lafand code as written).")
else:
    print("PATTERN: mixed/unclear - read the row-by-row tables above manually.")

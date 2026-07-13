#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:15:00
#SBATCH --job-name="diagnose-byt5-sentinels"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/diagnose_byt5_sentinels_%j.log
#SBATCH --error=logs/diagnose_byt5_sentinels_%j.log
#
# A/B diagnostic (no training) for the byt5 high-loss mystery.
# The ByT5 paper says sentinels are the FINAL 100 BYTE IDs (258 counting
# down), not added <extra_id_n> tokens. Our collator uses
# len(tokenizer)=384 as the sentinel base (383 counting down) - those
# rows of byt5's embedding matrix were never trained. This script runs
# the same batch through the model with both sentinel conventions (and
# with the paper's mean span length of 20 vs our 3.0) and prints the
# loss for each:
#   - 383-base ~8-12, 258-base ~2-6  -> sentinel bug confirmed
#   - both similar                   -> sentinels ruled out too
# Model runs in fp32 to keep dtype out of the comparison.

set -euo pipefail

export UV_LINK_MODE=copy

set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

uv run python3 - << 'EOF'
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.pretraining.collator import (
    create_sentinel_ids,
    filter_input_ids,
    random_spans_noise_mask,
    shift_tokens_right,
)

NOISE_DENSITY = 0.15
BATCH = 8

tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-large", torch_dtype=torch.float32)
model = model.to("cuda").eval()

dataset = load_from_disk("/scratch/rmdrak003/data/preprocessed/byt5")
input_ids = np.array([dataset[i]["input_ids"] for i in range(BATCH)])
length = input_ids.shape[1]
print(f"len(tokenizer)={len(tokenizer)}, chunk length={length}, batch={BATCH}")

# ByT5 vocab layout: ids 0-2 special, 3-258 the 256 bytes, 259-383 the
# HF-added extra_id tokens (embedding rows byt5 pretraining never used).
# sentinel base V means sentinels are V-1, V-2, ... counting down.
variants = [
    ("current collator: sentinel base 384 (ids 383 down), span 3.0", 384, 3.0),
    ("paper convention: sentinel base 259 (ids 258 down), span 3.0", 259, 3.0),
    ("paper convention + paper span: base 259, span 20.0", 259, 20.0),
]

for name, sentinel_base, span in variants:
    rng_state = np.random.RandomState(0)
    np.random.seed(0)  # same masks across variants so only sentinels/span differ
    mask = np.asarray([
        random_spans_noise_mask(length, NOISE_DENSITY, span) for _ in range(BATCH)
    ])
    labels_mask = ~mask

    input_sentinels = create_sentinel_ids(mask.astype(np.int8), sentinel_base)
    label_sentinels = create_sentinel_ids(labels_mask.astype(np.int8), sentinel_base)
    corrupted = filter_input_ids(input_ids, input_sentinels, tokenizer.eos_token_id)
    labels = filter_input_ids(input_ids, label_sentinels, tokenizer.eos_token_id)
    decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id, tokenizer.pad_token_id)

    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(corrupted).long().cuda(),
            labels=torch.from_numpy(labels).long().cuda(),
            decoder_input_ids=torch.from_numpy(decoder_input_ids).long().cuda(),
        )
    print(f"{name}\n  -> loss = {out.loss.item():.4f} (random baseline ~ ln(384) = 5.95)")
EOF

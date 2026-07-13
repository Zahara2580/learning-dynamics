#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:25:00
#SBATCH --job-name="diagnose-t5-nguni-sentinels"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/diagnose_t5_nguni_sentinels_%j.log
#SBATCH --error=logs/diagnose_t5_nguni_sentinels_%j.log
#
# Companion to diagnose_byt5_sentinels.sh, which confirmed the collator
# sentinel bug for byt5 (base 384: loss 8.09 worse-than-random; paper
# base 259: 4.88; base 259 + span 20: 0.81).
#
# This runs the same forward-loss A/B on the other two models:
#  - t5: sanity check that the current convention (base len(tokenizer)
#    = 32100, i.e. the real <extra_id_*> sentinels, span 3) is healthy.
#  - nguni-byt5: full 2x2 grid (sentinel base 384 vs 259, span 3 vs 20).
#    nguni-byt5 was itself CPT'd from byt5 by its authors, so whichever
#    cell gives the lowest loss reveals empirically which convention and
#    span its MAFT actually used - this decides what OUR collator must
#    use for nguni-byt5.
# fp32 throughout to keep dtype out of the comparison. No training.

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


def run_variants(model_name, data_path, variants):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to("cuda").eval()

    dataset = load_from_disk(data_path)
    input_ids = np.array([dataset[i]["input_ids"] for i in range(BATCH)])
    length = input_ids.shape[1]
    vocab_rows = model.get_input_embeddings().weight.shape[0]
    print(f"\n=== {model_name} ===")
    print(f"len(tokenizer)={len(tokenizer)}, embedding rows={vocab_rows}, chunk length={length}")

    for name, sentinel_base, span in variants:
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
        print(f"{name}\n  -> loss = {out.loss.item():.4f}")

    del model
    torch.cuda.empty_cache()


# t5: current convention is believed correct (<extra_id_*> really are the
# trained sentinels, counting down from len(tokenizer)-1 = 32099).
# Random baseline for t5's 32128-row output: ln(32128) ~ 10.38.
run_variants(
    "google-t5/t5-large",
    "/scratch/rmdrak003/data/preprocessed/t5",
    [
        ("current collator: sentinel base len(tok)=32100 (<extra_id_*>), span 3.0", 32100, 3.0),
    ],
)

# nguni-byt5: full 2x2 grid to empirically determine which convention its
# MAFT used. Random baseline ~ ln(384) = 5.95.
run_variants(
    "francois-meyer/nguni-byt5-large",
    "/scratch/rmdrak003/data/preprocessed/nguni-byt5",
    [
        ("A: sentinel base 384 (ids 383 down), span 3.0  [current collator]", 384, 3.0),
        ("B: sentinel base 384 (ids 383 down), span 20.0", 384, 20.0),
        ("C: sentinel base 259 (ids 258 down), span 3.0  [byt5 paper]", 259, 3.0),
        ("D: sentinel base 259 (ids 258 down), span 20.0 [byt5 paper + span]", 259, 20.0),
    ],
)
EOF

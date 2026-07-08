#!/bin/bash
# Tokenize MT (FLORES-200) or D2T (T2X) finetuning data for a given model config.
# Usage: scripts/preprocess_finetune_data.sh <mt|d2t> <t5|byt5|nguni-byt5> [nproc]
set -e

TASK="$1"
MODEL="$2"
NPROC="${3:-4}"

if [ -z "${TASK}" ] || [ -z "${MODEL}" ]; then
    echo "Usage: $0 <mt|d2t> <t5|byt5|nguni-byt5> [nproc]"
    exit 1
fi

uv run python3 -m src.data_processing.preprocess_finetune_data \
    --task "${TASK}" \
    --input "/scratch/rmdrak003/data/evaluation/${TASK}" \
    --model-config "configs/models/${MODEL}.yaml" \
    --output "/scratch/rmdrak003/data/preprocessed/finetune/${TASK}/${MODEL}" \
    --nproc "${NPROC}"

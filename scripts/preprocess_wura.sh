#!/bin/bash
# Tokenize and chunk the WURA corpus for a given model config.
# Usage: scripts/preprocess_wura.sh <t5|byt5|nguni-byt5> [nproc]
set -e

MODEL="$1"
NPROC="${2:-4}"

if [ -z "${MODEL}" ]; then
    echo "Usage: $0 <t5|byt5|nguni-byt5> [nproc]"
    exit 1
fi

uv run python3 -m src.data_processing.preprocess_wura \
    --input /scratch/rmdrak003/data/corpus/xho \
    --model-config "configs/models/${MODEL}.yaml" \
    --output "/scratch/rmdrak003/data/preprocessed/${MODEL}" \
    --nproc "${NPROC}"

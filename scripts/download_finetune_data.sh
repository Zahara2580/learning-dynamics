#!/bin/bash
# Download MT (FLORES-200) and D2T (T2X) evaluation/finetuning datasets.
# Usage: scripts/download_finetune_data.sh [mt|d2t|"mt d2t"] [language]
set -e

TASKS="${1:-mt d2t}"
LANGUAGE="${2:-xho}"

uv run python3 -m src.data_processing.download_finetune_data \
    --tasks ${TASKS} \
    --language "${LANGUAGE}" \
    --output-dir /scratch/rmdrak003/data/evaluation

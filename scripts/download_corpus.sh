#!/bin/bash
# Download the WURA corpus for CPT.
# Usage: scripts/download_corpus.sh [language]  (default: xho)
set -e

LANGUAGE="${1:-xho}"

uv run python3 -m src.data_processing.download_corpus \
    --output-dir /scratch/rmdrak003/data/corpus \
    --language "${LANGUAGE}"

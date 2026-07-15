#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --job-name="lafand-smoke-byt5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lafand_smoke_byt5_%j.log
#SBATCH --error=logs/lafand_smoke_byt5_%j.log
#
# End-to-end smoke test for byt5 on the lafand pipeline: confirms the
# byte-model code path works (ascending sentinel 259, byte tokenizer,
# variable-length collation, loads + trains + evals) before committing
# production GPU time. 60 steps, --no-save, effective batch 1024.
#
# Watch for: "first sentinel"/"Span" is preprocess-only, but here confirm
# the model loads at bf16, eval_loss is finite and declining, and the
# GPU-mem log line stays well under 48GB (lafand batches are
# variable-length, so peak differs from the old fixed-512 runs).
#
# Prerequisite (sintx CPU): lafand_preprocess for byt5 must have made
# /scratch/rmdrak003/data/lafand/byt5/train.* and dev.*

# Update to latest commit
git pull
git log -1

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# Load environment variables from .env
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

# HPC paths
export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.lafand_pretraining.continued_pretrain_lafand \
    --model-config configs/models/byt5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand/byt5 \
    --max-steps 60 \
    --warmup-steps 60 \
    --batch-size 8 \
    --gradient-accumulation-steps 128 \
    --eval-steps 15 \
    --n-eval-obs 2000 \
    --log-memory-every 5 \
    --wandb-run-name byt5-lafand-smoke \
    --metrics-filename metrics_lafand_smoke_byt5.jsonl \
    --run-subdir lafand-smoke \
    --no-save

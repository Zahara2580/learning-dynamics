#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:20:00
#SBATCH --job-name="cpt-fit-byt5-bs4-fp32"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/memory_fit_byt5_bs4_fp32_%j.log
#SBATCH --error=logs/memory_fit_byt5_bs4_fp32_%j.log
#
# bf16 vs fp32 DIAGNOSTIC (fp32 half) for byt5, at bs=4/accum=256 (effective
# 1024) - the earlier fp32 attempt at bs=8 OOM'd, so bs=4 here isolates
# dtype as the only variable against fitcheck_byt5_bs4_bf16.sh (same
# batch/accum, only --mixed_precision/--model-dtype differ). Slightly
# longer time limit since fp32 steps run slower than bf16.
# Compare the loss/grad_norm here against the bf16 run:
#   - both ~2-6, similar magnitude  -> bf16 was not the cause
#   - this one much lower than bf16 -> confirms bf16 weight rounding
# --no-save as this is a throwaway diagnostic.

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
    --mixed_precision no \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain \
    --model-config configs/models/byt5.yaml \
    --input /scratch/rmdrak003/data/preprocessed/byt5 \
    --eval-input /scratch/rmdrak003/data/preprocessed/byt5-validation \
    --max-steps 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 256 \
    --model-dtype fp32 \
    --wandb-run-name byt5-xho-fitcheck-bs4-fp32 \
    --metrics-filename metrics_fitcheck_byt5-bs4-fp32.jsonl \
    --run-subdir fitcheck-byt5-bs4-fp32 \
    --no-save

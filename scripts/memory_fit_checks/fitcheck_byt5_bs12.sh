#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --job-name="cpt-fit-byt5-bs12"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/memory_fit_byt5_bs12_%j.log
#SBATCH --error=logs/memory_fit_byt5_bs12_%j.log
#
# Quick memory-fit check for byt5: per_device_batch_size=12,
# gradient_accumulation_steps=85 (effective batch size 1020 - 12 doesn't divide 1024 evenly, closest divisor used).
# Only 3 steps, purely to check whether bs=12 fits in GPU memory on
# an L40S - not for loss/eval curves. bs=24 and bs=20 both OOM'd for this
# model (t5's ceiling was bs=20), so working down from bs=16.
# --no-save is passed since this is a throwaway fit check.

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
    --module src.pretraining.continued_pretrain \
    --model-config configs/models/byt5.yaml \
    --input /scratch/rmdrak003/data/preprocessed/byt5 \
    --eval-input /scratch/rmdrak003/data/preprocessed/byt5-validation \
    --max-steps 3 \
    --batch-size 12 \
    --gradient-accumulation-steps 85 \
    --wandb-run-name byt5-xho-fitcheck-bs12 \
    --metrics-filename metrics_fitcheck_byt5-bs12.jsonl \
    --run-subdir fitcheck-byt5-bs12 \
    --no-save

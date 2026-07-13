#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name="cpt-batch-byt5-bs24"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/batch_trial_byt5_bs24_%j.log
#SBATCH --error=logs/batch_trial_byt5_bs24_%j.log
#
# Batch size trial for byt5: per_device_batch_size=24,
# gradient_accumulation_steps=43 (effective batch size 1032 - 24 doesn't divide 1024 evenly, closest is used to test whether bs=24 even fits).
# 200 steps with a 100-step warmup, to compare validation loss curves
# across batch sizes 8/16/24 at a fixed effective batch size. --no-save is passed since these trials are only for loss/eval curves, not for keeping the model.

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
    --max-steps 200 \
    --warmup-steps 100 \
    --batch-size 24 \
    --gradient-accumulation-steps 43 \
    --wandb-run-name byt5-xho-bs24 \
    --metrics-filename metrics_byt5-xho-bs24.jsonl \
    --run-subdir byt5-xho-bs24 \
    --no-save

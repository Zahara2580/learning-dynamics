#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --job-name="cpt-batch-nguni-byt5-bs16"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/batch_trial_nguni-byt5_bs16_%j.log
#SBATCH --error=logs/batch_trial_nguni-byt5_bs16_%j.log
#
# Batch size trial for nguni-byt5: per_device_batch_size=16,
# gradient_accumulation_steps=64 (effective batch size 1024).
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
    --model-config configs/models/nguni-byt5.yaml \
    --input /scratch/rmdrak003/data/preprocessed/nguni-byt5 \
    --eval-input /scratch/rmdrak003/data/preprocessed/nguni-byt5-validation \
    --max-steps 200 \
    --warmup-steps 100 \
    --batch-size 16 \
    --gradient-accumulation-steps 64 \
    --wandb-run-name nguni-byt5-xho-bs16 \
    --metrics-filename metrics_nguni-byt5-xho-bs16.jsonl \
    --no-save

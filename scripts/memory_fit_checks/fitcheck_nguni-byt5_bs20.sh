#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --job-name="cpt-fit-nguni-byt5-bs20"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/memory_fit_nguni-byt5_bs20_%j.log
#SBATCH --error=logs/memory_fit_nguni-byt5_bs20_%j.log
#
# Quick memory-fit check for nguni-byt5: per_device_batch_size=20,
# gradient_accumulation_steps=51 (effective batch size 1020
# - 20 doesn't divide 1024 evenly, closest divisor used). Only 3 steps,
# purely to check whether bs=20 fits in GPU memory on an L40S - not for
# loss/eval curves. --no-save is passed since this is a throwaway fit check.

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
    --max-steps 3 \
    --batch-size 20 \
    --gradient-accumulation-steps 51 \
    --wandb-run-name nguni-byt5-xho-fitcheck-bs20 \
    --metrics-filename metrics_fitcheck_nguni-byt5-bs20.jsonl \
    --run-subdir fitcheck-nguni-byt5-bs20 \
    --no-save

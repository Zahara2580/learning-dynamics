#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:45:00
#SBATCH --job-name="cpt-test-batch1024"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/test_batch_size_1024_%j.log
#SBATCH --error=logs/test_batch_size_1024_%j.log
#
# 100-step trial at per_device_batch_size=256, gradient_accumulation_steps=4
# (effective batch size 1024) to check GPU memory stays stable in bf16
# over a longer run rather than just the first few steps.

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
    --model-config configs/models/t5.yaml \
    --input /scratch/rmdrak003/data/preprocessed/t5 \
    --max-steps 100 \
    --batch-size 256 \
    --gradient-accumulation-steps 4 \
    --log-memory-every 10

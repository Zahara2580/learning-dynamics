#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-test-eval"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/test_eval_%j.log
#SBATCH --error=logs/test_eval_%j.log
#
# 20-step trial with eval_steps=5 to check the validation loop
# (--eval-input, eval_strategy="steps") runs correctly and eval_loss
# shows up in metrics.jsonl / wandb, before trusting it on a real run.

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
    --eval-input /scratch/rmdrak003/data/preprocessed/t5-validation \
    --max-steps 20 \
    --eval-steps 5 \
    --batch-size 16 \
    --gradient-accumulation-steps 64

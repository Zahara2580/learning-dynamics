#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --job-name="lafand-pad-minus100"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lafand_pad_minus100_%j.log
#SBATCH --error=logs/lafand_pad_minus100_%j.log
#
# Padding A/B test (-100 in labels, padding excluded from loss): decides whether the lafand pipeline keeps
# the original's label padding with pad_token_id (padding contributes to
# the loss, as it did for nguni-byt5's own training) or switches to -100
# (padding excluded - pure LM loss). Identical short t5 run to its twin
# except for --ignore-pad-in-labels; compare train/eval loss level,
# shape, and noise between the two wandb runs, then pick.
#
# Prerequisite (once, on sintx CPU): export_wura_lines + lafand_preprocess
# for t5 must have produced /scratch/rmdrak003/data/lafand/t5/train.* and dev.*

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
    --model-config configs/models/t5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand/t5 \
    --max-steps 60 \
    --warmup-steps 60 \
    --batch-size 16 \
    --gradient-accumulation-steps 64 \
    --eval-steps 15 \
    --n-eval-obs 2000 \
    --wandb-run-name t5-lafand-pad-minus100 \
    --metrics-filename metrics_lafand_pad_minus100.jsonl \
    --run-subdir lafand-pad-minus100 \
    --no-save \
    --ignore-pad-in-labels

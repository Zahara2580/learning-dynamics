#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:15:00
#SBATCH --job-name="cpt-fit-byt5-bs8-fp32"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/memory_fit_byt5_bs8_fp32_%j.log
#SBATCH --error=logs/memory_fit_byt5_bs8_fp32_%j.log
#
# fp32 DIAGNOSTIC for byt5: the bf16 fitcheck showed loss ~1593 summed
# over accum=128 (~12.4/batch) at near-zero learning rate - i.e. the
# freshly *loaded* model scores worse than uniform-random (ln(384)~5.95)
# on our data, before training has changed anything. This run loads the
# model in fp32 with all bf16 autocast disabled (--model-dtype fp32 +
# --mixed_precision no) but is otherwise identical:
#   - loss drops to a sane ~2-6  -> bf16 weight rounding is the culprit
#   - loss stays ~12             -> suspect the preprocessed byt5 data
# fp32 roughly doubles memory vs bf16, but bs=8 fits comfortably.
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
    --batch-size 8 \
    --gradient-accumulation-steps 128 \
    --model-dtype fp32 \
    --wandb-run-name byt5-xho-fitcheck-bs8-fp32 \
    --metrics-filename metrics_fitcheck_byt5-bs8-fp32.jsonl \
    --run-subdir fitcheck-byt5-bs8-fp32 \
    --no-save

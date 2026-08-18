#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --job-name="cpt-fp32-byt5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/cpt_fp32_byt5_%j.log
#SBATCH --error=logs/cpt_fp32_byt5_%j.log
#
# fp32 CPT arm. Phase 1 trained with bf16 PARAMETERS, so AdamW wrote updates
# into an 8-bit mantissa and most were rounded away: 99.83% of t5's embeddings
# and 86.81% of all its parameters never changed in 10k steps. Measured, with
# the mechanism proven in isolation, in notes/bf16_finding.md.
#
# This run is identical to phase 1 in every other respect - same data, same
# effective batch (1024), same lr/warmup/steps, same sortish sampler - so the
# two arms differ ONLY in parameter precision.
#
# --model-dtype fp32 makes the trainer set bf16=False, so accelerate is told
# --mixed_precision no to match; transformers errors on a mismatch. Without
# autocast the activations are fp32 too, roughly doubling activation memory:
# t5 ~18GB and byt5 ~30GB of the L40S's 48GB, so it fits. Checkpoints are
# saved in the training dtype, so this arm archives fp32 snapshots.
#
# Writes to a NEW run-subdir so the bf16 arm is never resumed or overwritten.
# 6h links + chaining: short jobs schedule far faster on a congested queue.
#   sbatch --export=ALL,CHAIN_JOBS=6 scripts/lafand_fp32/cpt_fp32_<model>.sh

# Update to latest commit
git pull
git log -1

CHAIN_JOBS=${CHAIN_JOBS:-0}
if [ "${CHAIN_JOBS}" -gt 0 ]; then
    echo "Queuing next chained job (CHAIN_JOBS remaining after this: $((CHAIN_JOBS - 1)))"
    sbatch --dependency=afterany:${SLURM_JOB_ID} \
        --export=ALL,CHAIN_JOBS=$((CHAIN_JOBS - 1)) \
        "$0"
fi

export UV_LINK_MODE=copy

set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

# --resume is always passed: harmless on first run, resumes the chain after.
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision no \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain_lafand \
    --model-config configs/models/byt5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand/byt5 \
    --batch-size 4 \
    --gradient-accumulation-steps 256 \
    --save-steps 50 \
    --eval-steps 200 \
    --n-eval-obs 2000 \
    --wandb-run-name byt5-xho-lafand-bs4-fp32 \
    --metrics-filename metrics_byt5_lafand_bs4_fp32.jsonl \
    --run-subdir lafand-bs4-fp32 \
    --model-dtype fp32 \
    --sortish-sampler \
    --resume

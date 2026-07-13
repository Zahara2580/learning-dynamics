#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="cpt-t5-bs20"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/pretrain_t5_bs20_%j.log
#SBATCH --error=logs/pretrain_t5_bs20_%j.log
#
# Production batch-size CANDIDATE for t5: per_device_batch_size=20,
# gradient_accumulation_steps=51 (effective batch size 1020).
# Real full run (total_steps/warmup_steps from configs/models/t5.yaml, no --max-steps
# override, no --no-save) - this is genuinely resumable, not a throwaway
# test, because switching batch size after resuming would silently
# invalidate the optimizer/scheduler state Trainer restores.
#
# Methodology: this is one of several batch-size candidates for t5,
# each launched as its own real run under run-subdir "t5-xho-bs20" so
# they never share output_dir/checkpoints/resume paths. Compare loss/
# eval curves across candidates around step ~1000 via wandb, then:
#   - let the winning candidate keep running (or resubmit this same
#     script with --resume, unchanged, to continue the chain)
#   - scancel the other candidates for this model - their step 0-1000
#     compute is spent, but nothing here was invalid or a hack.
#
# Chains across 24h SLURM jobs via CHAIN_JOBS, same pattern as pretrain_t5.sh.

# Update to latest commit
git pull
git log -1

# Number of additional 24h jobs to chain after this one, so the run can
# span the ~3 days a full CPT run needs despite the 24-48h wall-time cap.
# Defaults to 0 (no auto-chaining) - pass CHAIN_JOBS=N as an env var to sbatch
# once you know which candidate won, to auto-chain it to completion. Each resubmission
# decrements it until it reaches 0, at which point no further job is queued.
CHAIN_JOBS=${CHAIN_JOBS:-0}
if [ "${CHAIN_JOBS}" -gt 0 ]; then
    echo "Queuing next chained job (CHAIN_JOBS remaining after this: $((CHAIN_JOBS - 1)))"
    sbatch --dependency=afterany:${SLURM_JOB_ID} \
        --export=ALL,CHAIN_JOBS=$((CHAIN_JOBS - 1)) \
        "$0"
fi

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

# --resume is always passed: find_latest_checkpoint returns None when no
# checkpoint exists yet, so the first job in the chain just starts fresh.
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain \
    --model-config configs/models/t5.yaml \
    --input /scratch/rmdrak003/data/preprocessed/t5 \
    --eval-input /scratch/rmdrak003/data/preprocessed/t5-validation \
    --batch-size 20 \
    --gradient-accumulation-steps 51 \
    --wandb-run-name t5-xho-bs20 \
    --metrics-filename metrics_t5-xho-bs20.jsonl \
    --run-subdir t5-xho-bs20 \
    --eval-steps 200 \
    --save-steps 200 \
    --resume

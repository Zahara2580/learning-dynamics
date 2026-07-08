#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="cpt-pretrain-t5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/pretrain_t5_%j.log
#SBATCH --error=logs/pretrain_t5_%j.log

# Update to latest commit
git pull
git log -1

# Number of additional 24h jobs to chain after this one, so the run can
# span the ~3 days a full CPT run needs despite the 24-48h wall-time cap.
# Defaults to 4 on first submission (no CHAIN_JOBS set); each resubmission
# decrements it until it reaches 0, at which point no further job is queued.
CHAIN_JOBS=${CHAIN_JOBS:-4}
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
    --resume

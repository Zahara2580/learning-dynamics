#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --job-name="backup-ckpts"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/backup_%j.log
#SBATCH --error=logs/backup_%j.log
#
# Offsite backup of the 3 CPT winner checkpoint sets to private HF repos.
# Runs on a worker node (large uploads must not run on the head node).
# The GPU is requested only so the job schedules on this account - the
# upload uses none. Resumable: resubmit if it times out and it continues.
#
#   sbatch scripts/backup_checkpoints.sh                    # all 3 models
#   sbatch --export=ALL,MODELS="t5" scripts/backup_checkpoints.sh

git pull

export UV_LINK_MODE=copy
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a
export HF_HOME=/scratch/rmdrak003/hf

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

MODELS_FLAG=""
if [ -n "${MODELS:-}" ]; then
    MODELS_FLAG="--models ${MODELS}"
fi

uv run python3 -m src.utils.backup_checkpoints ${MODELS_FLAG}

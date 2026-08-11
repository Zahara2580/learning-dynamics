#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name="africomet"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/africomet_%j.log
#SBATCH --error=logs/africomet_%j.log
#
# AfriCOMET (masakhane/africomet-mtl) over every saved MT prediction file:
# finetuned sweeps (3-epoch, 5-epoch, xh->en) + zero-shot, all 3 models,
# one GPU. Scoring only - our models never generate here. Resumable: rerun
# after more arms finish and only new files are scored.
#
# COMET lives in its own venv so its dependency pins never touch the
# project lockfile.
#
#   sbatch scripts/diagnostics/africomet.sh

git pull
git log -1

export UV_LINK_MODE=copy
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a
export HF_HOME=/scratch/rmdrak003/hf

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics

set -e

COMET_ENV=/scratch/rmdrak003/envs/africomet
if [ ! -x "${COMET_ENV}/bin/python" ]; then
    uv venv "${COMET_ENV}" --python 3.11
fi
uv pip install --python "${COMET_ENV}/bin/python" "unbabel-comet>=2.2" datasets

"${COMET_ENV}/bin/python" scripts/diagnostics/africomet_score.py
echo "=== done $(date -Is) ==="

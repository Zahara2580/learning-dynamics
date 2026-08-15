#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name="ablation"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/ablation_%x_%j.log
#SBATCH --error=logs/ablation_%x_%j.log
#
# Data-size ablation sweep: one model x one ablation config, all 20
# checkpoints, both selections. Hyperparameters identical to the full-
# data arms except training-set size. Resume-safe: resubmit to continue.
#
#   sbatch -J abl-t5-d2t1000 scripts/finetune/ablation.sh t5 d2t_1000
#   sbatch -J abl-t5-mt10k   scripts/finetune/ablation.sh t5 mt_10k

MODEL="${1:?usage: sbatch -J <name> ablation.sh <t5|byt5|nguni-byt5> <config basename>}"
CFG="${2:?usage: sbatch -J <name> ablation.sh <t5|byt5|nguni-byt5> <config basename>}"

# The ablations measure the PHASE-1 (monolingual) checkpoints. Pinned
# explicitly: auto-discovery is ambiguous now that lafand-bilingual/
# exists alongside them.
case "${MODEL}" in
  t5)         CKPTS=/scratch/rmdrak003/results/t5/lafand-bs8/checkpoints ;;
  byt5)       CKPTS=/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints ;;
  nguni-byt5) CKPTS=/scratch/rmdrak003/results/nguni-byt5/lafand-bs4/checkpoints ;;
  *) echo "unknown model ${MODEL}"; exit 1 ;;
esac

git pull
git log -1

export UV_LINK_MODE=copy
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a
export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export WANDB_DIR=/scratch/rmdrak003/learning-dynamics
export WANDB_CACHE_DIR=/scratch/rmdrak003/wandb-cache

WANDB_FLAG=""
if [ "${WANDB:-1}" -eq 1 ]; then
    WANDB_FLAG="--wandb"
fi

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

set -e

echo "=== ablation ${CFG} : ${MODEL}  start $(date -Is) ==="

uv run python3 -m src.finetuning.run_finetune \
    --config "configs/finetune/${CFG}.yaml" \
    --model "${MODEL}" \
    --checkpoints-dir "${CKPTS}" \
    --seed 42 \
    ${WANDB_FLAG}

echo "=== end $(date -Is) ==="

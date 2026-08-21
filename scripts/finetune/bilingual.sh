#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/finetune_%x_%j.log
#SBATCH --error=logs/finetune_%x_%j.log
#
# Finetune the PHASE-2 (bilingual isiXhosa+English) checkpoints. Full data,
# no ablation: D2T on the whole T2X train set, MT on 50,000 WMT22 pairs
# en->xh. Same protocol as the phase-1 full runs, separate results dirs.
#
#   sbatch -J bil-byt5-d2t scripts/finetune/bilingual.sh byt5 d2t_bilingual
#   sbatch -J bil-byt5-mt  scripts/finetune/bilingual.sh byt5 mt_bilingual
#
# Resume-safe: completed (checkpoint, selection) pairs are skipped via
# results.jsonl, so resubmitting after a timeout continues where it stopped.

MODEL="${1:?usage: sbatch -J <name> bilingual.sh <t5|byt5|nguni-byt5> <config basename>}"
CFG="${2:?usage: sbatch -J <name> bilingual.sh <t5|byt5|nguni-byt5> <config basename>}"

# Pinned explicitly - auto-discovery is ambiguous with lafand-bs*/ and
# lafand-bilingual/ side by side under the same results root.
CKPTS="/scratch/rmdrak003/results/${MODEL}/lafand-bilingual/checkpoints"
if [ ! -d "${CKPTS}" ]; then
    echo "no bilingual checkpoints at ${CKPTS}"; exit 1
fi

# --model <name>-bilingual resolves to configs/models/<name>-bilingual.yaml and
# writes model="<name>-bilingual" into every results row and prediction
# filename, so the arm is self-identifying and can never be mistaken for phase 1.
MODEL_ID="${MODEL}-bilingual"

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

echo "=== bilingual ${CFG} : ${MODEL_ID}  start $(date -Is) ==="
echo "    checkpoints: ${CKPTS}"

uv run python3 -m src.finetuning.run_finetune \
    --config "configs/finetune/${CFG}.yaml" \
    --model "${MODEL_ID}" \
    --checkpoints-dir "${CKPTS}" \
    --seed 42 \
    ${WANDB_FLAG}

echo "=== end $(date -Is) ==="

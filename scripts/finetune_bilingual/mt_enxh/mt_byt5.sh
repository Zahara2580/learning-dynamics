#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name="bil-mt-byt5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/bil_mt_byt5_%j.log
#SBATCH --error=logs/bil_mt_byt5_%j.log

MODEL=byt5
CFG=mt_bilingual
CKPTS="/scratch/rmdrak003/results/${MODEL}/lafand-bilingual/checkpoints"

if [ ! -d "${CKPTS}" ]; then
    echo "no bilingual checkpoints at ${CKPTS}"; exit 1
fi
N_CKPT=$(ls -d "${CKPTS}"/checkpoint-* 2>/dev/null | wc -l)
if [ "${N_CKPT}" -ne 19 ]; then
    echo "expected 19 checkpoints, found ${N_CKPT} "; exit 1
fi

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

echo "bilingual mt : ${MODEL} start $(date -Is)"


uv run python3 -m src.finetuning.run_finetune \
    --config "configs/finetune/${CFG}.yaml" \
    --model "${MODEL}-bilingual" \
    --checkpoints-dir "${CKPTS}" \
    --seed 42 \
    ${WANDB_FLAG}

echo "end $(date -Is)"

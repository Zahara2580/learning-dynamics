#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="cpt-bil-byt5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/bilingual_byt5_%j.log
#SBATCH --error=logs/bilingual_byt5_%j.log
#
# Phase-2 BILINGUAL CPT for byt5: same protocol as phase 1 (10k steps,
# same schedule, bs4 x accum 256 = effective 1024) on the passage-parity
# xho+eng corpus. New run-subdir + metrics file - never pooled with the
# monolingual run.
#
#   sbatch --export=ALL,CHAIN_JOBS=3 scripts/lafand_bilingual/cpt_byt5.sh

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

uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain_lafand \
    --model-config configs/models/byt5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand-bilingual/byt5 \
    --batch-size 4 \
    --gradient-accumulation-steps 256 \
    --save-steps 50 \
    --eval-steps 200 \
    --n-eval-obs 2000 \
    --wandb-run-name byt5-bilingual-lafand \
    --metrics-filename metrics_byt5_lafand_bilingual.jsonl \
    --run-subdir lafand-bilingual \
    --sortish-sampler \
    --eval-sets xho eng \
    --resume

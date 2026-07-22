#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --job-name="cpt-lafand-t5-bs8"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lafand_t5_bs8_%j.log
#SBATCH --error=logs/lafand_t5_bs8_%j.log
#
# Batch-size CANDIDATE for t5 on the lafand pipeline (thesis
# hyperparameter-tuning sweep): per_device_batch_size=8, accumulation
# 128 (effective 1024). Real full 10k-step run from step 0 - compare
# eval curves across candidates at ~step 1000 on wandb, keep the winner
# (its steps already count toward the 10k), scancel the others.
#
# WINNER (t5 sweep, decided at step ~3000: eval curves of bs4/bs8/bs16
# within 0.02 of each other and inside one eval-interval's improvement;
# bs8 had the highest measured steps/hour and the lowest eval loss of
# the two fast configs). Runs as 6h links - short jobs are allocated
# much faster on the congested l40sfree queue than 48h ones. Chain more
# links if needed: sbatch --export=ALL,CHAIN_JOBS=N <this script>
#
# Label padding is always -100 (excluded from loss) - decided via the
# padding A/B and confirmed by supervisor.

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
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain_lafand \
    --model-config configs/models/t5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand/t5 \
    --batch-size 8 \
    --gradient-accumulation-steps 128 \
    --save-steps 50 \
    --eval-steps 200 \
    --n-eval-obs 2000 \
    --wandb-run-name t5-xho-lafand-bs8 \
    --metrics-filename metrics_t5_lafand_bs8.jsonl \
    --run-subdir lafand-bs8 \
    --sortish-sampler \
    --resume

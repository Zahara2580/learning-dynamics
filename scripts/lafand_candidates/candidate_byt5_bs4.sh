#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="cpt-lafand-byt5-bs4"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lafand_byt5_bs4_%j.log
#SBATCH --error=logs/lafand_byt5_bs4_%j.log
#
# Batch-size CANDIDATE for byt5 on the lafand pipeline (thesis
# hyperparameter-tuning sweep): per_device_batch_size=4, accumulation
# 256 (effective 1024). Real full 10k-step run from step 0 - compare
# eval curves across candidates at ~step 1000 on wandb, keep the winner
# (its steps already count toward the 10k), scancel the others.
#
# CHAIN_JOBS defaults to 0 (single 24h job). To chain the WINNER to
# completion: sbatch --export=ALL,CHAIN_JOBS=6 <this script>
#
# Padding mode is faithful (pad-in-loss) unless --ignore-pad-in-labels
# is added below - settle via the padding A/B before submitting.

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
    --module src.lafand_pretraining.continued_pretrain_lafand \
    --model-config configs/models/byt5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand/byt5 \
    --batch-size 4 \
    --gradient-accumulation-steps 256 \
    --save-steps 50 \
    --eval-steps 200 \
    --n-eval-obs 2000 \
    --wandb-run-name byt5-xho-lafand-bs4 \
    --metrics-filename metrics_byt5_lafand_bs4.jsonl \
    --run-subdir lafand-bs4 \
    --sortish-sampler \
    --ignore-pad-in-labels \
    --resume

#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="cpt-lafand-nguni-byt5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lafand_prod_nguni-byt5_%j.log
#SBATCH --error=logs/lafand_prod_nguni-byt5_%j.log
#
# PRODUCTION lafand-pipeline CPT for nguni-byt5: full 10k steps / 5000 warmup /
# effective batch 1024 (the published Nguni-ByT5 recipe) on the offline
# lafand-corrupted paragraph data. Resume checkpoint every 50 steps
# (~1h of byte-model compute max lost on preemption); weights-only
# schedule checkpoints at the 19 two-phase steps for fine-tuning.
#
# BEFORE FIRST SUBMISSION: confirm batch-size 8/accum 128 against the
# GPU smoke's peak-memory log; batch must not change once the run starts.
# Padding mode is faithful (pad-in-loss) unless --ignore-pad-in-labels is
# added below - decide from the padding A/B before first submission.
#
# Chains additional 24h jobs via CHAIN_JOBS (default 4 resubmissions).

# Update to latest commit
git pull
git log -1

CHAIN_JOBS=${CHAIN_JOBS:-4}
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

# --resume is always passed: find_latest_checkpoint returns None when no
# checkpoint exists yet, so the first job in the chain starts fresh.
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.lafand_pretraining.continued_pretrain_lafand \
    --model-config configs/models/nguni-byt5.yaml \
    --data-dir /scratch/rmdrak003/data/lafand/nguni-byt5 \
    --batch-size 8 \
    --gradient-accumulation-steps 128 \
    --save-steps 50 \
    --eval-steps 200 \
    --n-eval-obs 2000 \
    --wandb-run-name nguni-byt5-xho-lafand \
    --metrics-filename metrics_nguni-byt5_lafand.jsonl \
    --run-subdir lafand-production \
    --resume

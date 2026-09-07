#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name="seedrep-d2t1000"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/seedrep_%x_%j.log
#SBATCH --error=logs/seedrep_%x_%j.log
#
# Seed-repeat experiment, LIMITED-DATA arm: byt5, D2T, 1,000 training
# examples, checkpoints {base, 5000, 10000} x seeds {42, 123, 456}.
#
# Protocol is configs/finetune/d2t_1000.yaml UNCHANGED - the same file that
# produced ablation_results/d2t_1000.jsonl (config_hash 649b6fe995e0).
# Only --seed and --results-dir vary. Nothing is tuned per checkpoint.
#
# The 1,000 examples are held FIXED across all seeds: the subset is drawn
# by random.Random(42) inside run_finetune, independent of --seed.
# verify_subset.py pins that down and aborts if it ever moves.
#
# Runtime measured from the existing byt5 1,000-example runs: mean 372s,
# max 412s per checkpoint. 9 runs -> ~56 min. 24h requested for headroom.
#
#   sbatch scripts/seed_experiment/d2t_1000_seeds.sh

MODEL=byt5
CONFIG=configs/finetune/d2t_1000.yaml
STEPS="0 5000 10000"
SEEDS=${SEEDS:-"42 123 456"}
FIRST=1
ARM=d2t_1000

# Phase-1 (monolingual isiXhosa) checkpoints were DELETED from /scratch, so
# they are restored from their HuggingFace backup below before training.
# This directory is created by fetch_checkpoints.py, not by the CPT run.
CKPTS=/scratch/rmdrak003/hf_ckpts/byt5

git pull
git log -1

export UV_LINK_MODE=copy
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a
export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"
export WANDB_DIR=/scratch/rmdrak003/learning-dynamics
export WANDB_CACHE_DIR=/scratch/rmdrak003/wandb-cache

# W&B off by default: 9 extra runs would clutter the main model-task groups.
WANDB_FLAG=""
if [ "${WANDB:-0}" -eq 1 ]; then
    WANDB_FLAG="--wandb"
fi

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

set -e

echo "=========================================================="
echo "seed repeat : ${MODEL} d2t 1,000 examples"
echo "  steps  : ${STEPS}"
echo "  seeds  : ${SEEDS}"
echo "  config : ${CONFIG}"
echo "  ckpts  : ${CKPTS}"
echo "  start  : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="

# Pin the training subset before anything trains. Creates the record on
# first run, then verifies and ABORTS on any drift.
uv run python3 -m scripts.seed_experiment.verify_subset --write

for STEP in 5000 10000; do
    echo ""
    echo "=== fetching checkpoint-${STEP} ==="
    uv run python3 -m scripts.seed_experiment.fetch_checkpoints \
        --model "${MODEL}" --steps "${STEP}" --dest "${CKPTS}"
    du -sh "${CKPTS}"

    # Step 0 is the un-adapted base model and needs no checkpoint of its own,
    # but run_finetune discovers checkpoints before filtering by --steps and
    # refuses an empty directory. So it rides along with the first fetch.
    if [ "${FIRST}" -eq 1 ]; then
        for SEED in ${SEEDS}; do
            OUT="seed_experiment/${ARM}_seed${SEED}"
            echo "--- step 0, seed ${SEED} -> ${OUT} ---"
            uv run python3 -m src.finetuning.run_finetune \
                --config "${CONFIG}" \
                --model "${MODEL}" \
                --checkpoints-dir "${CKPTS}" \
                --steps 0 \
                --seed "${SEED}" \
                --results-dir "${OUT}" \
                ${WANDB_FLAG}
        done
        FIRST=0
    fi

    for SEED in ${SEEDS}; do
        OUT="seed_experiment/${ARM}_seed${SEED}"
        echo "--- step ${STEP}, seed ${SEED} -> ${OUT} ---"
        uv run python3 -m src.finetuning.run_finetune \
            --config "${CONFIG}" \
            --model "${MODEL}" \
            --checkpoints-dir "${CKPTS}" \
            --steps "${STEP}" \
            --seed "${SEED}" \
            --results-dir "${OUT}" \
            ${WANDB_FLAG}
    done

    # Delete the source weights the moment every seed that needs them is done.
    # Scratch cannot hold both checkpoints at once, so this is required, not
    # tidiness. Cost of the trade: checkpoint-5000 and -10000 are each
    # downloaded once, and re-downloaded if the job is resubmitted.
    rm -rf "${CKPTS}/checkpoint-${STEP}"
    echo "=== deleted checkpoint-${STEP} ==="
    du -sh "${CKPTS}" 2>/dev/null || echo "  (checkpoint dir now empty)"
done

echo "=== end $(date -Is) ==="

uv run python3 -m src.finetuning.seed_analysis --arm "${ARM}" || \
    echo "(analysis skipped: the other arm may not be finished yet)"

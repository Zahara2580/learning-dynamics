#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name="seedrep-d2tfull"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/seedrep_%x_%j.log
#SBATCH --error=logs/seedrep_%x_%j.log
#
# Seed-repeat experiment, FULL-DATA arm: byt5, D2T, all 3,859 training
# examples, checkpoints {base, 5000, 10000} x seeds {42, 123, 456}.
#
# Protocol is configs/finetune/d2t_warmup20.yaml UNCHANGED - the same file
# that produced ablation_results/d2t_full.jsonl (config_hash 04715bd042d2).
# Only --seed and --results-dir vary. Nothing is tuned per checkpoint.
#
# Same train/valid/test files as the limited-data arm; the only difference
# between the two arms is that this one does not subsample the train split.
#
# Runtime measured from the existing byt5 full-data runs: mean 1,100s,
# max 1,184s per checkpoint. 9 runs -> ~2.8h. 24h requested for headroom.
#
#   sbatch scripts/seed_experiment/d2t_full_seeds.sh

MODEL=byt5
CONFIG=configs/finetune/d2t_warmup20.yaml
STEPS="0 5000 10000"
SEEDS=${SEEDS:-"42 123 456"}
ARM=d2t_full

# Phase-1 (monolingual isiXhosa) checkpoints, pinned: auto-discovery is
# ambiguous now that lafand-bilingual/ sits alongside them.
CKPTS=/scratch/rmdrak003/results/byt5/lafand-bs4/checkpoints

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
echo "seed repeat : ${MODEL} d2t FULL 3,859 examples"
echo "  steps  : ${STEPS}"
echo "  seeds  : ${SEEDS}"
echo "  config : ${CONFIG}"
echo "  ckpts  : ${CKPTS}"
echo "  start  : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="

# Asserts the train split is still 3,859 examples, which is what makes the
# two arms comparable. Also creates/verifies the limited-data subset record
# so both arms agree on the data even if only one has run.
uv run python3 -m scripts.seed_experiment.verify_subset --write

for SEED in ${SEEDS}; do
    OUT="seed_experiment/${ARM}_seed${SEED}"
    echo ""
    echo "--- seed ${SEED} -> ${OUT} ---"
    uv run python3 -m src.finetuning.run_finetune \
        --config "${CONFIG}" \
        --model "${MODEL}" \
        --checkpoints-dir "${CKPTS}" \
        --steps ${STEPS} \
        --seed "${SEED}" \
        --results-dir "${OUT}" \
        ${WANDB_FLAG}
done

echo "=== end $(date -Is) ==="

uv run python3 -m src.finetuning.seed_analysis --arm "${ARM}" || \
    echo "(analysis skipped: the other arm may not be finished yet)"

#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name="bil-d2t1000"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/bil_d2t1000_%j.log
#SBATCH --error=logs/bil_d2t1000_%j.log
#
# PHASE-2 (bilingual) D2T data-size ablation: 1,000 training examples,
# all three models, all 20 checkpoints. The bilingual counterpart of
# ablation_results/d2t_1000.jsonl.
#
# Protocol is byte-identical to the monolingual d2t_1000 arm - verified:
# configs/finetune/d2t_1000_bilingual.yaml hashes to 649b6fe995e0, the
# same as d2t_1000.yaml, because only results_dir and wandb_project
# differ and neither is in the hash.
#
# SAME 1,000 EXAMPLES as the monolingual arm. The subsample is drawn by
# random.Random(42) inside run_finetune from the same 3,859-line train
# split, independent of model, arm and --seed. Verified locally: the two
# configs load byte-identical train sources and targets.
#
# One job for all three models, ~2h each from the measured monolingual
# runtimes (372s x 20 checkpoints). 24h requested; leaves the second L40S
# free for the seed experiment.
#
# Resume-safe: completed (checkpoint, selection) pairs are skipped via
# results.jsonl, so resubmit after a timeout and it continues.
#
#   sbatch scripts/finetune_bilingual/d2t1000_all.sh
#   MODELS="byt5" sbatch scripts/finetune_bilingual/d2t1000_all.sh

MODELS=${MODELS:-"t5 byt5 nguni-byt5"}
CFG=d2t_1000_bilingual

# Fail before requesting anything if a checkpoint set is missing or partial.
for MODEL in ${MODELS}; do
    CKPTS="/scratch/rmdrak003/results/${MODEL}/lafand-bilingual/checkpoints"
    if [ ! -d "${CKPTS}" ]; then
        echo "no bilingual checkpoints at ${CKPTS}"; exit 1
    fi
    N_CKPT=$(ls -d "${CKPTS}"/checkpoint-* 2>/dev/null | wc -l)
    if [ "${N_CKPT}" -ne 19 ]; then
        echo "${MODEL}: expected 19 checkpoints, found ${N_CKPT} - refusing a partial sweep"
        exit 1
    fi
done

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

echo "=========================================================="
echo "bilingual d2t ablation : 1,000 examples"
echo "  models : ${MODELS}"
echo "  config : configs/finetune/${CFG}.yaml"
echo "  start  : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="

# Confirms the 1,000-example draw still matches the recorded one, so this
# arm trains on exactly the examples the monolingual arm used.
uv run python3 -m scripts.seed_experiment.verify_subset --write

for MODEL in ${MODELS}; do
    CKPTS="/scratch/rmdrak003/results/${MODEL}/lafand-bilingual/checkpoints"
    echo ""
    echo "--- ${MODEL}-bilingual  ($(date -Is)) ---"
    uv run python3 -m src.finetuning.run_finetune \
        --config "configs/finetune/${CFG}.yaml" \
        --model "${MODEL}-bilingual" \
        --checkpoints-dir "${CKPTS}" \
        --seed 42 \
        ${WANDB_FLAG}
done

echo "=== end $(date -Is) ==="
wc -l results_bilingual_d2t_1000/results.jsonl

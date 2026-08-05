#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name="zero-shot"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/zeroshot_%j.log
#SBATCH --error=logs/zeroshot_%j.log
#
# Zero-shot loss of base + all 19 checkpoints on the D2T test pairs, for
# all three models, without any finetuning. Direct behavioural evidence
# of whether CPT changed each model. ~60 model loads, loss only - well
# inside 6h on one L40S.
#
#   sbatch scripts/diagnostics/zero_shot_eval.sh

git pull
git log -1

export UV_LINK_MODE=copy
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a
export HF_HOME=/scratch/rmdrak003/hf

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

set -e

for MODEL in t5 byt5 nguni-byt5; do
    echo "=========================================================="
    echo "zero-shot ${MODEL} / d2t   $(date -Is)"
    echo "=========================================================="
    uv run python3 -m scripts.diagnostics.zero_shot_eval --model "${MODEL}" --task d2t
done

echo "=== done $(date -Is) ==="

#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name="zeroshot-mt"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/zeroshot_mt_%j.log
#SBATCH --error=logs/zeroshot_mt_%j.log
#
# Zero-shot MT (FLORES devtest, frozen generation settings) for base +
# all 19 checkpoints of ONE model. 1,012 sources is ~2.7x the D2T test
# set, so one model per job to stay inside 6h. Second arg picks the
# direction task: mt (en->xh, default) or mt-xhen (xh->en).
#
#   sbatch scripts/diagnostics/zero_shot_mt.sh t5
#   sbatch scripts/diagnostics/zero_shot_mt.sh byt5 mt-xhen

MODEL="${1:?usage: sbatch zero_shot_mt.sh <t5|byt5|nguni-byt5> [mt|mt-xhen]}"
TASK="${2:-mt}"

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

echo "=== zero-shot ${MODEL} / ${TASK}  $(date -Is) ==="
uv run python3 -m scripts.diagnostics.zero_shot_eval --model "${MODEL}" --task "${TASK}" "${@:3}"
echo "=== done $(date -Is) ==="

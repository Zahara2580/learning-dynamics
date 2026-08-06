#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name="xling-align"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/align_%j.log
#SBATCH --error=logs/align_%j.log
#
# Cross-lingual encoder alignment (eng vs xho FLORES dev pairs) for base
# + all 19 checkpoints, all three models. Encoder-only forward passes -
# well under the 4h limit.
#
#   sbatch scripts/diagnostics/crosslingual_alignment.sh

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
    echo "=== alignment ${MODEL}  $(date -Is) ==="
    uv run python3 -m scripts.diagnostics.crosslingual_alignment --model "${MODEL}"
done
echo "=== done $(date -Is) ==="

#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name="layer-align"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/layeralign_%j.log
#SBATCH --error=logs/layeralign_%j.log
#
# Layer-wise cross-lingual alignment: 5 evenly spaced checkpoints x every
# encoder layer, all three models. Encoder-only forward passes with
# hidden states - well inside 4h.
#
#   sbatch scripts/diagnostics/layerwise_alignment.sh

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
    echo "=== layer-wise alignment ${MODEL}  $(date -Is) ==="
    uv run python3 -m scripts.diagnostics.layerwise_alignment --model "${MODEL}"
done
echo "=== done $(date -Is) ==="

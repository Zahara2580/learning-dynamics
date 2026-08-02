#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name="wu20-d2t-t5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/warmup20_%x_%j.log
#SBATCH --error=logs/warmup20_%x_%j.log
#
# D2T for t5 with 20% warmup (one epoch to peak LR, then linear
# decay). Each checkpoint is trained once and evaluated twice - at the
# best-validation epoch and at the final epoch - so the two selection
# strategies are compared without training twice.
#
# Separate results dir and W&B project: a different protocol from the
# no-warmup sweep, and must never be pooled with it.
#
#   sbatch scripts/finetune/warmup20_d2t_t5.sh

MODEL=t5

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

WANDB_FLAG=""
if [ "${WANDB:-1}" -eq 1 ]; then
    WANDB_FLAG="--wandb"
fi

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

set -e

echo "=========================================================="
echo "d2t + 20% warmup : ${MODEL}   start $(date -Is)"
echo "=========================================================="

uv run python3 -m src.finetuning.run_finetune \
    --config configs/finetune/d2t_warmup20.yaml \
    --model "${MODEL}" \
    --seed 42 \
    ${WANDB_FLAG}

echo "=== end $(date -Is) ==="

# How often best-epoch and last-epoch actually differ - the question this
# experiment exists to answer.
uv run python3 -c "
import json, pathlib, collections
p = pathlib.Path('results_d2t_warmup_20/results.jsonl')
if p.exists():
    by = collections.defaultdict(dict); same = 0
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r['model'] != '${MODEL}': continue
        by[r['ckpt_step']][r['selection']] = r['metrics']['chrf']
        same += r.get('best_is_last', False) and r['selection'] == 'best_epoch'
    print(f\"{'step':>7} {'best':>8} {'last':>8} {'diff':>7}\")
    for step in sorted(by):
        d = by[step]
        if len(d) == 2:
            print(f\"{step:>7} {d['best_epoch']:>8.2f} {d['last_epoch']:>8.2f} \"
                  f\"{d['last_epoch']-d['best_epoch']:>+7.2f}\")
    print(f'checkpoints where best == last: {same}/{len(by)}')
"

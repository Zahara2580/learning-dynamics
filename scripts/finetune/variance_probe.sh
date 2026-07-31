#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name="variance-probe"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/variance_%x_%j.log
#SBATCH --error=logs/variance_%x_%j.log
#
# Seed-variance probe: re-runs a few checkpoints under extra seeds to
# measure how much of the spread between checkpoints is seed noise.
# Writes to results/variance/ so it can never be mixed with the sweep.
# Same protocol (identical config_hash) - only the seed differs.
#
#   sbatch scripts/finetune/variance_probe.sh              # t5 d2t, seeds 1 2 3
#   sbatch scripts/finetune/variance_probe.sh byt5 d2t
#   sbatch --export=ALL,SEEDS="1 2 3 4 5" scripts/finetune/variance_probe.sh

MODEL=${1:-t5}
TASK=${2:-d2t}
STEPS=${STEPS:-"0 5000 10000"}
SEEDS=${SEEDS:-"1 2 3"}
RESULTS_DIR=results/variance

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

# W&B off by default here: 9 extra seed runs would clutter the main
# model-task groups. WANDB=1 to enable.
WANDB_FLAG=""
if [ "${WANDB:-0}" -eq 1 ]; then
    WANDB_FLAG="--wandb"
fi

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

set -e

echo "=========================================================="
echo "variance probe: ${MODEL} / ${TASK}"
echo "  steps : ${STEPS}"
echo "  seeds : ${SEEDS}"
echo "  out   : ${RESULTS_DIR}"
echo "  start : $(date -Is)"
echo "=========================================================="

for SEED in ${SEEDS}; do
    echo "--- seed ${SEED} ---"
    uv run python3 -m src.finetuning.run_finetune \
        --config configs/finetune/${TASK}.yaml \
        --model "${MODEL}" \
        --steps ${STEPS} \
        --seed "${SEED}" \
        --results-dir "${RESULTS_DIR}" \
        ${WANDB_FLAG}
done

echo "=== end $(date -Is) ==="

# Spread per checkpoint across seeds - the number this job exists for.
uv run python3 -c "
import json, statistics, collections, pathlib
p = pathlib.Path('${RESULTS_DIR}/results.jsonl')
if not p.exists():
    raise SystemExit('no variance rows written')
by = collections.defaultdict(list)
for line in p.read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    if r['model'] == '${MODEL}' and r['task'] == '${TASK}':
        by[r['ckpt_step']].append((r['seed'], r['metrics']['chrf']))
print(f\"{'step':>7} {'n':>3} {'mean':>7} {'sd':>6} {'min':>7} {'max':>7} {'range':>6}\")
for step in sorted(by):
    vals = [v for _, v in by[step]]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f'{step:>7} {len(vals):>3} {statistics.mean(vals):>7.2f} {sd:>6.2f} '
          f'{min(vals):>7.2f} {max(vals):>7.2f} {max(vals)-min(vals):>6.2f}')
"

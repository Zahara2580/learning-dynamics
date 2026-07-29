#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --job-name="pilot-d2t-nguni-byt5"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/finetune_%x_%j.log
#SBATCH --error=logs/finetune_%x_%j.log
#
# PILOT: nguni-byt5 on d2t, base + final checkpoint ONLY (2 finetunes).
# Shakes out the pipeline and measures per-checkpoint wall-clock so the
# full 48h job can be budgeted. 6h: short jobs allocate faster.

MODEL=nguni-byt5
TASK=d2t

git pull
git log -1

CHAIN_JOBS=${CHAIN_JOBS:-0}
if [ "${CHAIN_JOBS}" -gt 0 ]; then
    echo "Queuing next chained job (remaining after this: $((CHAIN_JOBS - 1)))"
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

# W&B run files on scratch, not the home quota.
export WANDB_DIR=/scratch/rmdrak003/learning-dynamics
export WANDB_CACHE_DIR=/scratch/rmdrak003/wandb-cache

# W&B on by default (live curves, grouped nguni-byt5-d2t); WANDB=0 disables.
WANDB_FLAG=""
if [ "${WANDB:-1}" -eq 1 ]; then
    WANDB_FLAG="--wandb"
fi

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

# Without this a crashed run still exits 0 and SLURM reports COMPLETED.
set -e

echo "=========================================================="
echo "finetune ${MODEL} on ${TASK}  (PILOT base+final)   start $(date -Is)"
echo "=========================================================="

uv run python3 -m src.finetuning.run_finetune \
    --config configs/finetune/${TASK}.yaml \
    --model "${MODEL}" \
    --seed 42 \
    --pilot ${WANDB_FLAG}

echo "=== end $(date -Is) ==="

echo "Per-checkpoint runtimes recorded so far:"
uv run python3 -c "
import json, pathlib
p = pathlib.Path('results/finetune/results.jsonl')
if p.exists():
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r['model'] == '${MODEL}' and r['task'] == '${TASK}':
            print(f\"  step {r['ckpt_step']:>6}  {r['total_runtime_s']:>8.0f}s  \"
                  f\"BLEU {r['metrics']['bleu']:6.2f}  chrF {r['metrics']['chrf']:6.2f}\")
"

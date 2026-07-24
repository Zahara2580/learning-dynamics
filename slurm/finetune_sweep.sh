#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="ft-sweep"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/finetune_%x_%j.log
#SBATCH --error=logs/finetune_%x_%j.log
#
# Downstream finetuning sweep: one job finetunes and scores EVERY
# checkpoint of ONE model on ONE task, sequentially.
#
# Usage:
#   sbatch --job-name=ft-t5-t2x slurm/finetune_sweep.sh t5 t2x
#   sbatch --job-name=ft-t5-t2x --export=ALL,PILOT=1 slurm/finetune_sweep.sh t5 t2x
#   sbatch --job-name=ft-t5-t2x --export=ALL,CHAIN_JOBS=2 slurm/finetune_sweep.sh t5 t2x
#
# Not a job array: array elements queue independently and this cluster
# runs ~24h behind, so 20 array tasks would take weeks. One long job that
# loops internally gets the whole sweep from a single allocation.
#
# Resubmission is ALWAYS safe. Completed runs are keyed in results.jsonl
# and skipped, so a job that dies at checkpoint 12 is fixed by
# resubmitting; a job that finds everything done exits in seconds.

MODEL=${1:?usage: finetune_sweep.sh <model> <task>}
TASK=${2:?usage: finetune_sweep.sh <model> <task>}

# Update to latest commit
git pull
git log -1

CHAIN_JOBS=${CHAIN_JOBS:-0}
if [ "${CHAIN_JOBS}" -gt 0 ]; then
    echo "Queuing next chained job (CHAIN_JOBS remaining after this: $((CHAIN_JOBS - 1)))"
    sbatch --dependency=afterany:${SLURM_JOB_ID} \
        --job-name="${SLURM_JOB_NAME}" \
        --export=ALL,CHAIN_JOBS=$((CHAIN_JOBS - 1)) \
        "$0" "${MODEL}" "${TASK}"
fi

export UV_LINK_MODE=copy

set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

PILOT_FLAG=""
if [ "${PILOT:-0}" -eq 1 ]; then
    PILOT_FLAG="--pilot"
    echo "PILOT MODE: base model and final checkpoint only"
fi

echo "=========================================================="
echo "model : ${MODEL}"
echo "task  : ${TASK}"
echo "start : $(date -Is)"
echo "=========================================================="

uv run python3 -m src.finetuning.run_finetune \
    --config configs/finetune/${TASK}.yaml \
    --model "${MODEL}" \
    --seed 42 \
    ${PILOT_FLAG}

echo "=========================================================="
echo "end   : $(date -Is)"
echo "=========================================================="

# Per-checkpoint wall-clock, for budgeting the remaining jobs. The pilot
# needs this number to decide how many 48h links the full sweep takes.
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

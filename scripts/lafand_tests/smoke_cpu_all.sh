#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name="lafand-cpu-smoke"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/lafand_cpu_smoke_%j.log
#SBATCH --error=logs/lafand_cpu_smoke_%j.log
#
# CPU-only logic confirmation of the whole lafand pipeline while GPUs
# are queued: runs all three models (plus the -100 padding variant on
# t5) for 2 tiny optimizer steps each. Purpose is ONLY to prove the
# code path end-to-end (data loads, collator, forward/backward,
# optimizer step, eval loop, callbacks) - NOT loss behavior at scale:
# CPU is ~30-100x slower, so batch sizes here are tiny and the numbers
# are not comparable to the GPU smokes.
#
# fp32 (--model-dtype fp32): bf16 autocast is a GPU feature; the dtype
# check in the trainer enforces consistency either way.
#
# If sbatch rejects the partition/account, run the same four commands
# directly inside a sintx shell instead - they need no GPU.

# Update to latest commit
git pull
git log -1

export UV_LINK_MODE=copy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

run_smoke () {
    local name=$1 config=$2 data=$3 extra=$4
    echo ""
    echo "================ CPU smoke: ${name} ================"
    uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand \
        --model-config configs/models/${config} \
        --data-dir /scratch/rmdrak003/data/lafand/${data} \
        --model-dtype fp32 \
        --max-steps 2 \
        --warmup-steps 2 \
        --batch-size 2 \
        --gradient-accumulation-steps 2 \
        --eval-steps 2 \
        --n-eval-obs 16 \
        --wandb-run-name ${name}-lafand-cpusmoke \
        --metrics-filename metrics_cpusmoke_${name}.jsonl \
        --run-subdir lafand-cpusmoke \
        --no-save ${extra} \
        || { echo "FAILED: ${name}"; exit 1; }
    echo "PASSED: ${name}"
}

# t5 first (fastest - fails fastest if something is broken), then the
# -100 label-padding flag path, then both byte models.
run_smoke t5 t5.yaml t5 ""
run_smoke t5-minus100 t5.yaml t5 "--ignore-pad-in-labels"
run_smoke byt5 byt5.yaml byt5 ""
run_smoke nguni-byt5 nguni-byt5.yaml nguni-byt5 ""

echo ""
echo "================ ALL FOUR CPU SMOKES PASSED ================"

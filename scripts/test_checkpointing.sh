#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --job-name="cpt-test-checkpointing"
#SBATCH --mail-user=rmdrak003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/test_checkpointing_%j.log
#SBATCH --error=logs/test_checkpointing_%j.log
#
# Sanity-checks the checkpoints/ vs resume/ split and --resume behaviour
# for continued_pretrain.py, in one short GPU allocation, before trusting
# it on a real multi-day chained run:
#   1. Run for a handful of steps (fresh start), forcing a resume/
#      checkpoint to be saved early via --save-steps.
#   2. Run again with --resume for a few more steps.
#   3. Assert: resume/ has exactly one checkpoint-<N>/ with optimizer.pt
#      (proves save_total_limit pruning works), checkpoints/ has
#      weights-only folders with no optimizer.pt, and the second run's
#      logged global_step picks up after the first run's, not from 0.
#
# Usage: sbatch scripts/test_checkpointing.sh

set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-configs/models/t5.yaml}
INPUT=${INPUT:-/scratch/rmdrak003/data/preprocessed/t5}
EVAL_INPUT=${EVAL_INPUT:-/scratch/rmdrak003/data/preprocessed/t5-validation}
TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR:-/scratch/rmdrak003/results/test-checkpointing}
FIRST_RUN_STEPS=${FIRST_RUN_STEPS:-5}
SECOND_RUN_STEPS=${SECOND_RUN_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-3}

# Suppress uv hardlink warning
export UV_LINK_MODE=copy

# Load environment variables from .env
set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

# HPC paths
export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_HOME}"

# Load Python and sync dependencies
module load python/miniconda3-py3.12
cd /scratch/rmdrak003/learning-dynamics
uv sync --frozen

# Build a throwaway model config pointing at TEST_OUTPUT_DIR, so this
# test never writes into the real per-model output_dir from MODEL_CONFIG.
TEST_CONFIG=$(mktemp --suffix=.yaml)
uv run python3 - "$MODEL_CONFIG" "$TEST_OUTPUT_DIR" "$TEST_CONFIG" << 'EOF'
import sys
import yaml

src, output_dir, dest = sys.argv[1:4]
with open(src) as f:
    config = yaml.safe_load(f)
config["output_dir"] = output_dir
with open(dest, "w") as f:
    yaml.safe_dump(config, f)
EOF

echo "=== Test config written to ${TEST_CONFIG} (output_dir=${TEST_OUTPUT_DIR}) ==="
rm -rf "${TEST_OUTPUT_DIR}"

echo ""
echo "=== Run 1/2: fresh start, ${FIRST_RUN_STEPS} steps, save_steps=${SAVE_STEPS} ==="
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain \
    --model-config "${TEST_CONFIG}" \
    --input "${INPUT}" \
    --eval-input "${EVAL_INPUT}" \
    --max-steps "${FIRST_RUN_STEPS}" \
    --save-steps "${SAVE_STEPS}"

echo ""
echo "=== Run 1 checkpoint state ==="
echo "-- checkpoints/ (weights-only, expect NO optimizer.pt) --"
find "${TEST_OUTPUT_DIR}/checkpoints" -maxdepth 2
echo "-- resume/ (full-state, expect optimizer.pt) --"
find "${TEST_OUTPUT_DIR}/resume" -maxdepth 2

echo ""
echo "=== Run 2/2: --resume, extending to ${SECOND_RUN_STEPS} steps total ==="
uv run accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    --main_process_port $((29501 + SLURM_JOB_ID % 1000)) \
    --module src.pretraining.continued_pretrain \
    --model-config "${TEST_CONFIG}" \
    --input "${INPUT}" \
    --eval-input "${EVAL_INPUT}" \
    --max-steps "${SECOND_RUN_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --resume \
    2>&1 | tee "${TEST_OUTPUT_DIR}/run2.log"

echo ""
echo "=== Assertions ==="

# 1. resume/ should have exactly one checkpoint dir (save_total_limit=1 pruning).
RESUME_CKPT_COUNT=$(find "${TEST_OUTPUT_DIR}/resume" -maxdepth 1 -type d -name 'checkpoint-*' | wc -l)
if [ "${RESUME_CKPT_COUNT}" -ne 1 ]; then
    echo "FAIL: expected exactly 1 checkpoint under resume/, found ${RESUME_CKPT_COUNT}"
    exit 1
fi
echo "PASS: resume/ has exactly 1 checkpoint (save_total_limit=1 pruning works)"

# 2. That checkpoint must contain optimizer.pt (full state).
RESUME_CKPT_DIR=$(find "${TEST_OUTPUT_DIR}/resume" -maxdepth 1 -type d -name 'checkpoint-*')
if [ ! -f "${RESUME_CKPT_DIR}/optimizer.pt" ]; then
    echo "FAIL: ${RESUME_CKPT_DIR} is missing optimizer.pt"
    exit 1
fi
echo "PASS: ${RESUME_CKPT_DIR} contains optimizer.pt"

# 3. Weights-only checkpoints must NOT contain optimizer.pt.
for dir in "${TEST_OUTPUT_DIR}"/checkpoints/checkpoint-*/; do
    if [ -f "${dir}optimizer.pt" ]; then
        echo "FAIL: ${dir} unexpectedly contains optimizer.pt (should be weights-only)"
        exit 1
    fi
done
echo "PASS: checkpoints/ folders are weights-only (no optimizer.pt)"

# 4. Run 2's log should show it resumed from a step > 0, not started fresh.
if ! grep -q "Resuming from the latest" "${TEST_OUTPUT_DIR}/run2.log"; then
    echo "FAIL: run 2 log has no 'Resuming from the latest' message - resume_from was never set"
    exit 1
fi
echo "PASS: run 2 logged that it resumed from a saved checkpoint"

echo ""
echo "=== All checkpointing assertions passed ==="
echo "Test output left in ${TEST_OUTPUT_DIR} for manual inspection; delete when done:"
echo "  rm -rf ${TEST_OUTPUT_DIR}"

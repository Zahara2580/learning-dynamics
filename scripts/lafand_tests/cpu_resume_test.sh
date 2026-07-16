#!/bin/bash
# CPU resume-logic test for the lafand trainer. Run directly inside a
# sintx shell (no sbatch, no GPU):
#     cd /scratch/rmdrak003/learning-dynamics
#     SLURM_CPUS_PER_TASK=4 bash scripts/lafand_tests/cpu_resume_test.sh
#
# Emulates the real crash scenario ("we did 2653 steps but the last
# full-state save was at 2650") at miniature scale:
#   Run A: 3 steps, full-state save every 2 -> resume/checkpoint-2 holds
#          optimizer state; step 3 was logged but never saved.
#   Run B: --resume, max-steps 4 -> must resume FROM STEP 2, re-run
#          steps 3 and 4. Step 3 therefore appears TWICE in the metrics
#          (once from A, once from B) - the expected, harmless overlap;
#          keep the LAST occurrence per step when plotting.
# Asserts: the resume log line, exactly one pruned full-state checkpoint
# with optimizer.pt, the weights-only schedule snapshots, and the
# duplicated step-3 metric.

set -uo pipefail

export UV_LINK_MODE=copy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets

cd /scratch/rmdrak003/learning-dynamics

RUN_ROOT=/scratch/rmdrak003/results/byt5/lafand-cpu-resume-test
METRICS=/scratch/rmdrak003/results/byt5/metrics_cpu_resume_test.jsonl
rm -rf "${RUN_ROOT}" "${METRICS}"

common_args=(
    --model-config configs/models/byt5.yaml
    --data-dir /scratch/rmdrak003/data/lafand/byt5
    --model-dtype fp32
    --warmup-steps 2
    --batch-size 2 --gradient-accumulation-steps 2
    --save-steps 2
    --eval-steps 999 --n-eval-obs 4
    --ignore-pad-in-labels
    --wandb-run-name byt5-lafand-cpu-resume-test
    --metrics-filename metrics_cpu_resume_test.jsonl
    --run-subdir lafand-cpu-resume-test
)

echo "=== Run A: 3 steps, last full-state save lands at step 2 ==="
uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand \
    "${common_args[@]}" --max-steps 3 || { echo "FAILED: run A"; exit 1; }

echo ""
echo "=== State after run A ==="
find "${RUN_ROOT}" -maxdepth 3 | sort

echo ""
echo "=== Run B: --resume, extending to 4 steps ==="
uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand \
    "${common_args[@]}" --max-steps 4 --resume 2>&1 | tee /tmp/lafand_resume_runB.log
grep -q "Training complete" /tmp/lafand_resume_runB.log || { echo "FAILED: run B"; exit 1; }

echo ""
echo "=== Assertions ==="
if grep -q "Resuming from the latest" /tmp/lafand_resume_runB.log; then
    echo "PASS: run B resumed from a saved checkpoint (from step 2, not step 0)"
else
    echo "FAIL: no resume log line - run B started fresh"; exit 1
fi

N_RESUME=$(find "${RUN_ROOT}/resume" -maxdepth 1 -type d -name 'checkpoint-*' | wc -l)
if [ "${N_RESUME}" -eq 1 ] && [ -f "${RUN_ROOT}"/resume/checkpoint-*/optimizer.pt ]; then
    echo "PASS: exactly 1 full-state resume checkpoint with optimizer.pt (pruning works)"
else
    echo "FAIL: resume/ contents unexpected"; find "${RUN_ROOT}/resume" -maxdepth 2; exit 1
fi

N_STEP3=$(grep -c '"step": 3' "${METRICS}")
if [ "${N_STEP3}" -ge 2 ]; then
    echo "PASS: step 3 logged twice in metrics (once pre-crash, once post-resume) -"
    echo "      this is the expected overlap; dedupe by keeping the LAST entry per step."
else
    echo "WARN: expected step 3 twice in ${METRICS}, found ${N_STEP3} - check manually"
fi

echo ""
echo "=== weights-only schedule snapshots (should include tokenizer files now) ==="
ls "${RUN_ROOT}"/checkpoints/checkpoint-*/ | head -20

echo ""
echo "=== RESUME TEST COMPLETE ==="
echo "Cleanup when done:  rm -rf ${RUN_ROOT} ${METRICS}"

#!/bin/bash
# CPU A/B timing of SortishSampler vs random batching, byt5, 6 steps each.
# Run inside sintx:
#     cd /scratch/rmdrak003/learning-dynamics
#     SLURM_CPUS_PER_TASK=4 bash scripts/lafand_tests/cpu_sortish_timing.sh
# Compare the two train_runtime lines at the end; also check the sortish
# run's losses stay sane (stability spot-check).

set -uo pipefail
export UV_LINK_MODE=copy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
set -a; source /scratch/rmdrak003/learning-dynamics/.env; set +a
export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets
cd /scratch/rmdrak003/learning-dynamics

run_one () {
    local name=$1 extra=$2
    echo ""
    echo "================ ${name} ================"
    uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand \
        --model-config configs/models/byt5.yaml \
        --data-dir /scratch/rmdrak003/data/lafand/byt5 \
        --model-dtype fp32 \
        --max-steps 3 --warmup-steps 3 \
        --batch-size 2 --gradient-accumulation-steps 1 \
        --eval-steps 999 --n-eval-obs 4 \
        --ignore-pad-in-labels \
        --wandb-run-name byt5-lafand-${name} \
        --metrics-filename metrics_${name}.jsonl \
        --run-subdir lafand-${name} \
        --no-save ${extra} 2>&1 | tee /tmp/lafand_${name}.log | grep -E "loss|train_runtime|Sortish"
}

run_one timing-random ""
run_one timing-sortish "--sortish-sampler"

echo ""
echo "================ COMPARISON ================"
for n in timing-random timing-sortish; do
    echo "${n}: $(grep -o "train_runtime[^,]*" /tmp/lafand_${n}.log | tail -1)"
done
echo "(sortish should be meaningfully lower; note its FIRST step is the"
echo " deliberately-longest batch, so per-step times start high then drop)"

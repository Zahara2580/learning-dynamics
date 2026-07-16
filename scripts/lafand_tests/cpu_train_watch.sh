#!/bin/bash
# CPU mini-training curves for all three models on the lafand pipeline,
# while GPUs are queued. Run directly inside a sintx shell:
#     cd /scratch/rmdrak003/learning-dynamics
#     SLURM_CPUS_PER_TASK=4 bash scripts/lafand_tests/cpu_train_watch.sh
#
# NOT a performance indicator: fp32, effective batch 4 (vs 1024 in
# production), warmup compressed to fit the run - the loss will be far
# noisier and the LR schedule completely different. What it IS good for:
# watching that each model's loss engages (moves rather than exploding
# or NaNing) under the real objective, and eyeballing relative levels
# (nguni should start lowest - the pipeline is its native format).
#
# ~60s/step for the byte models on 4 CPU threads; t5 can be slower.
# Override steps per model:  WATCH_STEPS=6 bash ...

set -uo pipefail

export UV_LINK_MODE=copy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
STEPS=${WATCH_STEPS:-10}

set -a
source /scratch/rmdrak003/learning-dynamics/.env
set +a

export HF_HOME=/scratch/rmdrak003/hf
export HF_DATASETS_CACHE=${HF_HOME}/datasets

cd /scratch/rmdrak003/learning-dynamics

for m in byt5 nguni-byt5 t5; do
    echo ""
    echo "================ watch: $m (${STEPS} steps) ================"
    uv run python3 -m src.lafand_pretraining.continued_pretrain_lafand \
        --model-config configs/models/${m}.yaml \
        --data-dir /scratch/rmdrak003/data/lafand/${m} \
        --model-dtype fp32 \
        --max-steps "${STEPS}" \
        --warmup-steps "${STEPS}" \
        --batch-size 2 --gradient-accumulation-steps 2 \
        --eval-steps 5 --n-eval-obs 8 \
        --ignore-pad-in-labels \
        --wandb-run-name ${m}-lafand-cpuwatch \
        --metrics-filename metrics_cpuwatch_${m}.jsonl \
        --run-subdir lafand-cpuwatch \
        --no-save || { echo "FAILED: $m"; exit 1; }
    echo "DONE: $m"
done

echo ""
echo "All three watch runs complete - compare the *-lafand-cpuwatch runs on wandb."

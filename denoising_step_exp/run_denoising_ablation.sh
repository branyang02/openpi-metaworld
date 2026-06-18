#!/bin/bash
# Run denoising step ablation: evaluate pi0.5 with 1, 2, 3, 5, 10 Euler steps.
# Each step count runs as a separate OS process to avoid GPU memory fragmentation.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=1
#   bash denoising_step_exp/run_denoising_ablation.sh

set -euo pipefail

CHECKPOINT="checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"
CONFIG="pi05_metaworld"
SPLIT="train"
OUTPUT_DIR="denoising_step_exp/results/ablation"
STEP_COUNTS=(1 2 3 5 10)

echo "=== Denoising Step Ablation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Config: ${CONFIG}"
echo "Split: ${SPLIT}"
echo "Step counts: ${STEP_COUNTS[*]}"
echo ""

for NUM_STEPS in "${STEP_COUNTS[@]}"; do
    echo "--- Running num_steps=${NUM_STEPS} ---"
    MUJOCO_GL=egl uv run denoising_step_exp/eval_denoising_steps.py \
        --policy.config="${CONFIG}" \
        --policy.dir="${CHECKPOINT}" \
        --num_steps "${NUM_STEPS}" \
        --split "${SPLIT}" \
        --output_dir "${OUTPUT_DIR}"
    echo "--- Completed num_steps=${NUM_STEPS} ---"
    echo ""
done

echo "=== All runs complete ==="
echo "Results in: ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}"/results_*steps.json

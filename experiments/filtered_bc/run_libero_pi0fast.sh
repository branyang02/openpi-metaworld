#!/usr/bin/env bash
# Full libero_10 sweep for the pi0-FAST filtered-BC baseline on LIBERO.
#
# Mirrors run_libero.sh but swaps the base ckpt + configs for pi0-FAST. The
# subprocess flow is unchanged: each task spawns a pi0-FAST policy server in the
# root venv + a rollout client in examples/libero_env/'s Python 3.8 venv. After
# LoRA merge, a fresh server is spawned with the merged ckpt for eval. Total
# wall-clock per task is dominated by two cold policy-server startups.
#
# Launch with nohup so it survives terminal detach:
#     nohup bash experiments/filtered_bc/run_libero_pi0fast.sh \
#         > experiments/filtered_bc/logs/libero_pi0fast.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000}
RESULTS_JSON=${RESULTS_JSON:-experiments/filtered_bc/results_libero_pi0fast.json}
LIBERO_SUITE=${LIBERO_SUITE:-libero_10}

mkdir -p experiments/filtered_bc/logs

uv run python -u -m experiments.filtered_bc.run_filtered_bc \
    --args.env libero \
    --args.base-ckpt "$BASE_CKPT" \
    --args.base-config pi0_fast_libero \
    --args.train-config pi0_fast_libero_low_mem_finetune \
    --args.libero-suite "$LIBERO_SUITE" \
    --args.split train \
    --args.num-rollouts 30 \
    --args.num-train-steps 200 \
    --args.batch-size 8 \
    --args.eval-num-episodes 30 \
    --args.replan-steps 5 \
    --args.results-json "$RESULTS_JSON"

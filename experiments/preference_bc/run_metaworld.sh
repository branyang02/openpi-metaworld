#!/usr/bin/env bash
# 10-task MetaWorld sweep for the preference-BC (Flow-DPO) baseline.
# Task list mirrors experiments/filtered_bc/run_metaworld.sh so the two
# parametric baselines are evaluated on the same task set.
#
# Launch with nohup so it survives terminal detach:
#     nohup bash experiments/preference_bc/run_metaworld.sh \
#         > experiments/preference_bc/logs/metaworld.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-metaworld-5000}
RESULTS_JSON=${RESULTS_JSON:-experiments/preference_bc/results_metaworld.json}
BETA=${BETA:-2000.0}

mkdir -p experiments/preference_bc/logs

uv run python -u -m experiments.preference_bc.run_preference_bc \
    --args.env metaworld \
    --args.base-ckpt "$BASE_CKPT" \
    --args.tasks \
        coffee-push-v3 \
        push-v3 \
        pick-place-v3 \
        plate-slide-back-v3 \
        faucet-close-v3 \
        pick-place-wall-v3 \
        reach-v3 \
        coffee-pull-v3 \
        disassemble-v3 \
        stick-push-v3 \
    --args.num-rollouts 30 \
    --args.num-train-steps 200 \
    --args.batch-size 8 \
    --args.eval-num-episodes 30 \
    --args.beta "$BETA" \
    --args.results-json "$RESULTS_JSON"

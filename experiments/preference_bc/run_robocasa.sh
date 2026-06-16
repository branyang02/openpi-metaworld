#!/usr/bin/env bash
# Flow-DPO sweep on RoboCasa 7-task subset.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000}
RESULTS_JSON=${RESULTS_JSON:-experiments/preference_bc/results_robocasa.json}
ROBOCASA_TASK_SET=${ROBOCASA_TASK_SET:-subset}
BETA=${BETA:-2000.0}

mkdir -p experiments/preference_bc/logs

uv run python -u -m experiments.preference_bc.run_preference_bc \
    --args.env robocasa \
    --args.base-ckpt "$BASE_CKPT" \
    --args.robocasa-task-set "$ROBOCASA_TASK_SET" \
    --args.split train \
    --args.num-rollouts 30 \
    --args.num-train-steps 200 \
    --args.batch-size 8 \
    --args.eval-num-episodes 30 \
    --args.replan-steps 5 \
    --args.beta "$BETA" \
    --args.results-json "$RESULTS_JSON"

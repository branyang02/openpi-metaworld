#!/usr/bin/env bash
# 7-task subset sweep for the filtered-BC baseline on RoboCasa.
#
# Same subprocess orchestration as LIBERO (see run_libero.sh header). Each task
# rollout picks up task_horizon from robocasa.utils.dataset_registry_utils and
# runs for 1.5x that many steps.
#
# Launch with nohup so it survives terminal detach:
#     nohup bash experiments/filtered_bc/run_robocasa.sh \
#         > experiments/filtered_bc/logs/robocasa.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000}
RESULTS_JSON=${RESULTS_JSON:-experiments/filtered_bc/results_robocasa.json}
ROBOCASA_TASK_SET=${ROBOCASA_TASK_SET:-subset}

mkdir -p experiments/filtered_bc/logs

uv run python -u -m experiments.filtered_bc.run_filtered_bc \
    --args.env robocasa \
    --args.base-ckpt "$BASE_CKPT" \
    --args.robocasa-task-set "$ROBOCASA_TASK_SET" \
    --args.split train \
    --args.num-rollouts 30 \
    --args.num-train-steps 200 \
    --args.batch-size 8 \
    --args.eval-num-episodes 30 \
    --args.replan-steps 5 \
    --args.results-json "$RESULTS_JSON"

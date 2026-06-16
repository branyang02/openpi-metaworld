#!/usr/bin/env bash
# 7-task RoboCasa filtered-BC sweep for GR00T N1.5.
#
# Mirrors run_robocasa.sh (pi05) but routes the orchestrator at
# experiments/filtered_bc/groot/run_filtered_bc_groot.py — which launches
# groot_env/serve.py for both rollout and post-merge eval, and spawns the
# groot_env-based LoRA trainer between them.
#
# Launch with nohup so it survives terminal detach:
#     nohup bash experiments/filtered_bc/run_robocasa_groot.sh \
#         > experiments/filtered_bc/logs/robocasa_groot.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000}
RESULTS_JSON=${RESULTS_JSON:-experiments/filtered_bc/results_robocasa_groot.json}

mkdir -p experiments/filtered_bc/logs

uv run python -u -m experiments.filtered_bc.groot.run_filtered_bc_groot \
    --args.base-ckpt "$BASE_CKPT" \
    --args.num-rollouts 30 \
    --args.num-train-steps 200 \
    --args.batch-size 8 \
    --args.eval-num-episodes 30 \
    --args.replan-steps 5 \
    --args.results-json "$RESULTS_JSON"

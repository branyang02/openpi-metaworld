#!/usr/bin/env bash
# 10-task MetaWorld sweep for the pi0-FAST filtered-BC baseline.
#
# Mirrors run_metaworld.sh but swaps the base ckpt + configs for pi0-FAST.
# pi0-FAST is loaded as JAX in-process (the existing PyTorch converter only
# knows about pi0/pi0.5), and the merged eval policy is also reloaded as JAX
# from a scratch checkpoint dir — see run_filtered_bc.py:_run_one_task_inprocess.
#
# Launch with nohup so it survives terminal detach:
#     nohup bash experiments/filtered_bc/run_metaworld_pi0fast.sh \
#         > experiments/filtered_bc/logs/metaworld_pi0fast.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
# JAX must release GPU memory between the train and eval phases of each task;
# without these two, its pool holds the train-state allocation and the eval
# policy reload OOMs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500}
RESULTS_JSON=${RESULTS_JSON:-experiments/filtered_bc/results_metaworld_pi0fast.json}

mkdir -p experiments/filtered_bc/logs

uv run python -u -m experiments.filtered_bc.run_filtered_bc \
    --args.env metaworld \
    --args.base-ckpt "$BASE_CKPT" \
    --args.base-config pi0_fast_metaworld \
    --args.train-config pi0_fast_metaworld_low_mem_finetune \
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
    --args.results-json "$RESULTS_JSON"

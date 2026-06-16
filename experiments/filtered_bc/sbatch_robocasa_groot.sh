#!/usr/bin/env bash
# sbatch wrapper for the GR00T+RoboCasa filtered-BC sweep on dj-high.
#
# Direct usage:
#     sbatch -J fbc_groot_robocasa \
#         -o experiments/filtered_bc/logs/fbc_groot_robocasa-%j.out \
#         -e experiments/filtered_bc/logs/fbc_groot_robocasa-%j.err \
#         experiments/filtered_bc/sbatch_robocasa_groot.sh
#
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --time=24:00:00
#SBATCH --gpus=l40:1
#SBATCH --mem-per-gpu=80G
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
INNER="${REPO_ROOT}/experiments/filtered_bc/run_robocasa_groot.sh"
if [[ ! -x "$INNER" ]]; then
    echo "missing inner script: $INNER" >&2
    exit 2
fi
cd "$REPO_ROOT"

echo "[sbatch_robocasa_groot.sh] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[sbatch_robocasa_groot.sh] gpus visible: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

exec bash "$INNER"

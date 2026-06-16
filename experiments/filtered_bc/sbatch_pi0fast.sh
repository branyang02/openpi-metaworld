#!/usr/bin/env bash
# sbatch wrapper that runs the per-env pi0-FAST filtered-BC sweep on dj-high.
#
# Submit once per env via launch_pi0fast.sh.
# Direct usage:
#     sbatch -J fbc_pi0fast_metaworld \
#         -o experiments/filtered_bc/logs/fbc_pi0fast_metaworld-%j.out \
#         -e experiments/filtered_bc/logs/fbc_pi0fast_metaworld-%j.err \
#         experiments/filtered_bc/sbatch_pi0fast.sh metaworld
#
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high
#SBATCH --time=24:00:00
#SBATCH --gpus=l40:1
#SBATCH --mem-per-gpu=80G
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch ... $0 <metaworld|libero>" >&2
    exit 2
fi
ENV=$1

case "$ENV" in
    metaworld|libero) ;;
    *) echo "unknown env: $ENV (expected metaworld|libero)" >&2; exit 2 ;;
esac

# SLURM stages the sbatch script into /var/spool/slurmd/<job>/, so $0 isn't
# in the repo. Resolve via SLURM_SUBMIT_DIR (the cwd at sbatch-time, which
# launch_pi0fast.sh sets to the repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
INNER="${REPO_ROOT}/experiments/filtered_bc/run_${ENV}_pi0fast.sh"
if [[ ! -x "$INNER" ]]; then
    echo "missing inner script: $INNER" >&2
    exit 2
fi
cd "$REPO_ROOT"

echo "[sbatch_pi0fast.sh] host=$(hostname) job=${SLURM_JOB_ID:-?} env=${ENV}"
echo "[sbatch_pi0fast.sh] gpus visible: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

exec bash "$INNER"

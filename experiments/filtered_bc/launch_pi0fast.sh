#!/usr/bin/env bash
# Submit two sbatch jobs for the pi0-FAST filtered-BC sweep:
# - fbc_pi0fast_metaworld (10-task ML45 subset)
# - fbc_pi0fast_libero (full libero_10)
# Each lands its own L40 GPU on dineshj-compute / dj-high for 24h.
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p experiments/filtered_bc/logs

for ENV in metaworld libero; do
    sbatch \
        -J "fbc_pi0fast_${ENV}" \
        -o "experiments/filtered_bc/logs/fbc_pi0fast_${ENV}-%j.out" \
        -e "experiments/filtered_bc/logs/fbc_pi0fast_${ENV}-%j.err" \
        experiments/filtered_bc/sbatch_pi0fast.sh "$ENV"
done

echo
echo "Submitted. Track with:"
echo "    squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.6D %R'"
echo
echo "Stream logs (once a job starts running):"
echo "    tail -f experiments/filtered_bc/logs/fbc_pi0fast_metaworld-*.err"
echo "    tail -f experiments/filtered_bc/logs/fbc_pi0fast_libero-*.err"
echo
echo "Per-task results JSON files (incremental writes):"
echo "    experiments/filtered_bc/results_metaworld_pi0fast.json"
echo "    experiments/filtered_bc/results_libero_pi0fast.json"

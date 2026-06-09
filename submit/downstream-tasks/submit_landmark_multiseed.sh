#!/bin/bash

set -euo pipefail

LAUNCHER=./submit_landmark_1gpu.sh

for SEED in 0 1 2 3 4; do
    echo "===== submitting seed=$SEED ====="
    sbatch --export=ALL,SPLIT_SEED=$SEED "$LAUNCHER"
done

echo
echo "5 jobs queued. Aggregate with:"
echo "  python -m downstream_tasks.landmark._aggregate_multiseed \\"
echo "      --runs_root ./outputs \\"
echo "      --pattern   landmark_brainbench_openus_v1_seed{seed}"

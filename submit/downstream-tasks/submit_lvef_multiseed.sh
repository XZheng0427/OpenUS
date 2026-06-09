#!/bin/bash

set -euo pipefail

LAUNCHER=./submit_lvef_1gpu.sh

for SEED in 0 1 2 3 4; do
    echo "===== submitting seed=$SEED ====="
    sbatch --export=ALL,SPLIT_SEED=$SEED "$LAUNCHER"
done

echo
echo "5 jobs queued. Watch with: squeue -u \$USER"
echo "Once all finish, aggregate with:"
echo "  python -m downstream_tasks.landmark._aggregate_multiseed \\"
echo "      --runs_root ./outputs \\"
echo "      --pattern   lvef_camus_openus_seed{seed}"

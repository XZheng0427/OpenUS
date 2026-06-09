#!/bin/bash
# 5-seed sweep wrapper for downstream task 4 (cardiac detection).
# Same data split for every seed (fixed paper-aligned 250 / 50); seed
# varies only the model initialisation + sampler order.


set -euo pipefail

PROJECT_DIR=./OpenUS
LAUNCHER=$PROJECT_DIR/submit_cardiac_detection_1gpu.sh
PREPARED_DIR=$PROJECT_DIR/OpenUS_datasets/FOCUS-dataset/_prepared

if [[ ! -f "$PREPARED_DIR/.done" ]]; then
    echo "[error] $PREPARED_DIR/.done not found." >&2
    echo "        Run: bash $PROJECT_DIR/submit_cardiac_detection_prepare.sh" >&2
    echo "        once before submitting the multiseed sweep." >&2
    exit 1
fi

for SEED in 0 1 2 3 4; do
    echo "===== submitting seed=$SEED ====="
    sbatch --export=ALL,SPLIT_SEED=$SEED "$LAUNCHER"
done

echo
echo "5 jobs queued. Aggregate with:"
echo "  python -m downstream_tasks.cardiac_detection._aggregate_multiseed \\"
echo "      --runs_root $PROJECT_DIR/outputs \\"
echo "      --pattern   cardiac_focus_openus_v1_seed{seed}"

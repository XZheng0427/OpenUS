#!/bin/bash
# Prepare the FOCUS dataset for downstream-task 4 (cardiac detection).
# Run this ONCE before any submit_cardiac_detection_*.sh training jobs.
#
# It merges FOCUS training/ + validation/ -> _prepared/trainval/ (250 imgs),
# copies testing/ -> _prepared/test/ (50 imgs), writes a meta.json per split,
# and touches a .done marker. Multiseed training jobs ASSERT on the marker
# and bail out if it is missing.
#
# This is run interactively (no SBATCH) -- the work is < 30 s of file I/O
# and a few MB.

set -euo pipefail

PROJECT_DIR=./OpenUS
DATA_ROOT=$PROJECT_DIR/OpenUS_datasets/FOCUS-dataset
PREPARED_DIR=$DATA_ROOT/_prepared

source ./miniforge3/etc/profile.d/conda.sh
# prepare_focus has no mmrotate dependency; the base openus env is enough.
conda activate openus

cd "$PROJECT_DIR"
python -m downstream_tasks.cardiac_detection.prepare_focus \
    --data_root    "$DATA_ROOT" \
    --prepared_dir "$PREPARED_DIR" \
    "$@"

echo
echo "FOCUS dataset prepared at: $PREPARED_DIR"
echo "Marker:                     $PREPARED_DIR/.done"
echo
echo "Next:"
echo "  sbatch submit_cardiac_detection_1gpu.sh"
echo "  bash   submit_cardiac_detection_multiseed.sh"

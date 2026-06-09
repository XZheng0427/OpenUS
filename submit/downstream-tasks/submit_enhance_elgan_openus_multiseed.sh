#!/bin/bash
# US-DINO encoder inside the EnlightenGAN harness (enlightengan_vmamba),
# 5-seed grid at the chosen encoder policy and resolution.


set -euo pipefail

LAUNCHER=./submit_enhance_enlightengan_1gpu.sh
FREEZE_ENCODER=${FREEZE_ENCODER:-False}
IMAGE_SIZE=${IMAGE_SIZE:-256}

for SEED in 0 1 2 3 4; do
    echo "===== submitting vmamba seed=$SEED freeze=$FREEZE_ENCODER img=$IMAGE_SIZE ====="
    sbatch --export=ALL,SPLIT_SEED=$SEED,GENERATOR=vmamba,FREEZE_ENCODER=$FREEZE_ENCODER,IMAGE_SIZE=$IMAGE_SIZE "$LAUNCHER"
done

POL=$([[ "$FREEZE_ENCODER" == "True" ]] && echo frozen || echo finetune)
echo
echo "5 jobs queued. Output dirs: enlightengan_vmamba_seed{0..4}_${POL}_img${IMAGE_SIZE}"

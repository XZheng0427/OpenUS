#!/bin/bash
#SBATCH --job-name=enlightengan_enhance_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=workq
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/enhance_enlightengan_1gpu_%j.out
#SBATCH --error=./logs/enhance_enlightengan_1gpu_%j.err


set -euo pipefail

PROJECT_DIR=./OpenUS
SPLIT_SEED=${SPLIT_SEED:-42}
IMAGE_SIZE=${IMAGE_SIZE:-256}
GENERATOR=${GENERATOR:-vmamba}
FREEZE_ENCODER=${FREEZE_ENCODER:-False}

VSSM_INIT=$PROJECT_DIR/pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth
OPENUS_CKPT=./models/openus_cpt0150.pth

POL=$([[ "$FREEZE_ENCODER" == "True" ]] && echo frozen || echo finetune)
POL_OFF=$([[ "$FREEZE_ENCODER" == "True" ]] && echo 0 || echo 1)
IMG_OFF=$([[ "$IMAGE_SIZE" == "256" ]] && echo 0 || echo 500)

RUN_NAME=enlightengan_vmamba_seed${SPLIT_SEED}_${POL}_img${IMAGE_SIZE}
GEN_ARGS="--generator vmamba --pretrained_vmamba_init $VSSM_INIT --pretrained_weights $OPENUS_CKPT --checkpoint_key teacher --freeze_encoder $FREEZE_ENCODER --encoder_lr_scale 0.1"
PORT=$((31100 + 2 * SPLIT_SEED + POL_OFF + IMG_OFF))

OUTPUT_DIR=$PROJECT_DIR/outputs/$RUN_NAME

DATA_ROOT=$PROJECT_DIR/OpenUS_datasets/image_enhancement
MANIFEST_PATH=$DATA_ROOT/enhance_manifest_seed${SPLIT_SEED}.json
HOLDOUT_MANIFEST=$DATA_ROOT/enhance_holdout.json

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

module load libfabric/1.22.0
module load craype-network-ofi
module load brics/aws-ofi-nccl/1.8.1
export FI_PROVIDER=cxi

export MASTER_PORT=$PORT
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/tmp/${USER}/triton_${SLURM_JOB_ID}

echo "===== Job setup ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "SPLIT_SEED=$SPLIT_SEED  IMAGE_SIZE=$IMAGE_SIZE  GENERATOR=$GENERATOR  FREEZE_ENCODER=$FREEZE_ENCODER  PORT=$PORT"

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "[head] Building enhance manifest (seed=$SPLIT_SEED)..."
    source ./miniforge3/etc/profile.d/conda.sh
    conda activate openus
    cd "$PROJECT_DIR"
    python -m downstream_tasks.enhance.prepare_enhance \
        --data_root "$DATA_ROOT" --seed $SPLIT_SEED \
        --out "$MANIFEST_PATH" --holdout_out "$HOLDOUT_MANIFEST"
    conda deactivate
fi

source ./miniforge3/etc/profile.d/conda.sh
conda activate openus
python - <<'PY' 2>/dev/null || pip install --quiet pyiqa clean-fid
import pyiqa, cleanfid
PY
conda deactivate

srun --export=ALL --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    bash -lc '
        set -euo pipefail
        source ./miniforge3/etc/profile.d/conda.sh
        conda activate openus

        module load libfabric/1.22.0 craype-network-ofi brics/aws-ofi-nccl/1.8.1
        export FI_PROVIDER=cxi
        export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/${USER}/triton_${SLURM_JOB_ID}}
        mkdir -p "$TRITON_CACHE_DIR"

        cd '"$PROJECT_DIR"'
        export PYTHONPATH='"$PROJECT_DIR"':${PYTHONPATH:-}

        echo "[node] $(hostname) GPUs=$CUDA_VISIBLE_DEVICES"

        python eval_enhance_enlightengan.py \
            '"$GEN_ARGS"' \
            --data_root '"$DATA_ROOT"' \
            --manifest_path '"$MANIFEST_PATH"' \
            --holdout_manifest '"$HOLDOUT_MANIFEST"' \
            --image_size '"$IMAGE_SIZE"' \
            --use_norm True \
            --batch_size 8 \
            --num_workers 4 \
            --lr 1e-4 \
            --beta1 0.5 \
            --niter 100 \
            --niter_decay 100 \
            --patch_size 32 \
            --num_extra_patches 5 \
            --vgg_weight 1.0 \
            --seed '"$SPLIT_SEED"' \
            --output_dir '"$OUTPUT_DIR"'
    '

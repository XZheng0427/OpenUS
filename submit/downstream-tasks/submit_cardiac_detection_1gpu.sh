#!/bin/bash
#SBATCH --job-name=openus_cardiac_det_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=workq
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=./logs/cardiac_det_1gpu_%j.out
#SBATCH --error=./logs/cardiac_det_1gpu_%j.err

# Downstream task 4: Rotated Faster R-CNN with OpenUS-VMamba backbone on
# FOCUS (fetal cardiac four-chamber view).

set -euo pipefail

PROJECT_DIR=./OpenUS
SPLIT_SEED=${SPLIT_SEED:-42}
RUN_NAME=cardiac_focus_openus_v1_seed${SPLIT_SEED}
OUTPUT_DIR=$PROJECT_DIR/outputs/$RUN_NAME

DATA_ROOT=$PROJECT_DIR/OpenUS_datasets/FOCUS-dataset
PREPARED_DIR=$DATA_ROOT/_prepared

VSSM_INIT=$PROJECT_DIR/pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth
OPENUS_CKPT=./models/openus_cpt0150.pth

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

if [[ ! -f "$PREPARED_DIR/.done" ]]; then
    echo "[error] $PREPARED_DIR/.done not found." >&2
    echo "        Run: bash $PROJECT_DIR/submit_cardiac_detection_prepare.sh" >&2
    echo "        before submitting any training jobs." >&2
    exit 1
fi

module load libfabric/1.22.0
module load craype-network-ofi
module load brics/aws-ofi-nccl/1.8.1
export FI_PROVIDER=cxi
export NCCL_BUFFSIZE=4194304
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1

# MASTER_PORT base 29700 for the cardiac-detection task family.
export MASTER_PORT=$((29700 + SPLIT_SEED))

export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0
MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{print $1}')
if [[ -z "$MASTER_ADDR" ]]; then
  export MASTER_ADDR="$MASTER_HOSTNAME"
fi

export TRITON_CACHE_DIR=/tmp/${USER}/triton_${SLURM_JOB_ID}

echo "===== Job setup ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_NNODES=$SLURM_NNODES"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PREPARED_DIR=$PREPARED_DIR"

srun --export=ALL --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    bash -lc '
        set -euo pipefail
        source ./miniforge3/etc/profile.d/conda.sh
        # mmrotate stack lives in `openus-mmrot` (clone of `openus` + mim install).
        conda activate openus-mmrot

        module load libfabric/1.22.0 craype-network-ofi brics/aws-ofi-nccl/1.8.1
        export FI_PROVIDER=cxi

        export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/${USER}/triton_${SLURM_JOB_ID}}
        mkdir -p "$TRITON_CACHE_DIR"

        # Force TMPDIR to a per-job HOME-based dir. The cluster default
        # TMPDIR=/local/user/<UID>/ is unreliable on Grace Hopper nodes -- 
        # it disappears between mkdir and torchrun on some nodes 
        # (seen on nid011229, nid011169). HOME is shared / persistent,
        # so this always works. Per-job path avoids cross-job collisions.
        export TMPDIR=$HOME/.cardiac_tmp/job_${SLURM_JOB_ID}
        mkdir -p "$TMPDIR"
        echo "[node $SLURM_NODEID] TMPDIR=$TMPDIR (forced to HOME path)"

        cd '"$PROJECT_DIR"'
        export PYTHONPATH='"$PROJECT_DIR"':${PYTHONPATH:-}
        export NODE_RANK=$SLURM_NODEID

        echo "[node $SLURM_NODEID/$SLURM_NNODES] $(hostname) GPUs=$CUDA_VISIBLE_DEVICES MASTER=$MASTER_ADDR:$MASTER_PORT"

        torchrun \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            eval_cardiac_detection.py \
                --arch vmamba_small \
                --vmamba_imagenet_ckpt '"$VSSM_INIT"' \
                --pretrained_weights '"$OPENUS_CKPT"' \
                --checkpoint_key teacher \
                --loss_type smooth_l1 \
                --ctr_score_threshold 0.3 \
                --ctr_tolerances "0.03,0.05,0.10" \
                --batch_size_per_gpu 8 \
                --num_workers 4 \
                --epochs 100 \
                --lr 1e-4 \
                --seed '"$SPLIT_SEED"' \
                --data_root    '"$DATA_ROOT"' \
                --prepared_dir '"$PREPARED_DIR"' \
                --output_dir   '"$OUTPUT_DIR"'
    '

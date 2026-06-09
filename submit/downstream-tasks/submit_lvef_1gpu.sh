#!/bin/bash
#SBATCH --job-name=usdino_lvef_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=workq
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=./logs/lvef_1gpu_%j.out
#SBATCH --error=./logs/lvef_1gpu_%j.err

set -euo pipefail

PROJECT_DIR=./OpenUS
SPLIT_SEED=${SPLIT_SEED:-0}
RUN_NAME=lvef_camus_openus_seed${SPLIT_SEED}
OUTPUT_DIR=$PROJECT_DIR/outputs/$RUN_NAME

CAMUS_ROOT=./OpenUS2/prepared_datasets/CAMUS_2
CAMUS_MANIFEST=$CAMUS_ROOT/camus_lvef_manifest_seed${SPLIT_SEED}.json

VSSM_INIT=$PROJECT_DIR/pretrained/vmamba/vssm_small_0229_ckpt_epoch_222.pth
OPENUS_CKPT=./models/openus_cpt0150.pth

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

module load libfabric/1.22.0
module load craype-network-ofi
module load brics/aws-ofi-nccl/1.8.1
export FI_PROVIDER=cxi
export NCCL_BUFFSIZE=4194304
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1

export MASTER_PORT=$((29620 + SPLIT_SEED))

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
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CAMUS_MANIFEST=$CAMUS_MANIFEST"

if [[ ! -f "$CAMUS_MANIFEST" ]]; then
    echo "[head] Building CAMUS LVEF manifest (seed=$SPLIT_SEED)..."
    source ./miniforge3/etc/profile.d/conda.sh
    conda activate openus
    cd "$PROJECT_DIR"
    python -m downstream_tasks.lvef.prepare_camus \
        --data_root  "$CAMUS_ROOT" \
        --out        "$CAMUS_MANIFEST" \
        --split_seed $SPLIT_SEED
    conda deactivate
fi

srun --export=ALL --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
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
        export NODE_RANK=$SLURM_NODEID

        echo "[node $SLURM_NODEID/$SLURM_NNODES] $(hostname) GPUs=$CUDA_VISIBLE_DEVICES MASTER=$MASTER_ADDR:$MASTER_PORT"

        torchrun \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            eval_lvef.py \
                --arch vmamba_small \
                --pretrained_vmamba_init '"$VSSM_INIT"' \
                --pretrained_weights '"$OPENUS_CKPT"' \
                --checkpoint_key teacher \
                --img_size 224 \
                --batch_size_per_gpu 16 \
                --num_workers 4 \
                --epochs 100 \
                --lr 1e-4 \
                --weight_decay 0.05 \
                --hidden_dim 512 \
                --dropout 0.1 \
                --loss smooth_l1 \
                --enable_vflip False \
                --enable_jitter True \
                --camus_manifest '"$CAMUS_MANIFEST"' \
                --images_root '"$CAMUS_ROOT"' \
                --output_dir '"$OUTPUT_DIR"'
    '

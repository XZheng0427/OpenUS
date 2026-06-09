#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --mem=0 
#SBATCH --job-name=validation_cls_2_busi_from100_130
#SBATCH --output=./logs/slurm_validation_cls_vmamba_small_2_busi_from100_130_%j.out
#SBATCH --error=./logs/slurm_validation_cls_vmamba_small_2_busi_from100_130_%j.err

# Initialize conda and activate environment
source ~/miniforge3/bin/activate
module load cuda/12.2
conda activate vmamba-arm

PROJECT_DIR="./OpenUS"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
echo 'The work dir is: ' $PROJECT_DIR

# Set distributed training environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0

# Set distributed training environment variables for single-GPU setup
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29509
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0


LR_VALUES=(0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)
AVGPOOL_VALUES=(0 1 2)
cpk_values=(0100 0110 0120 0130)
for cpk in "${cpk_values[@]}"; do
    for LR in "${LR_VALUES[@]}"; do
        for AVGPOOL in "${AVGPOOL_VALUES[@]}"; do
            echo "Running with LR=$LR and avgpool_patchtokens=$AVGPOOL"
            
            python eval_linear.py \
            --arch vmamba_small \
            --num_labels 6 \
            --data_path './BUSI/BUSI_split' \
            --pretrained_weights "./OpenUS/results/vmamba_small_4gpu_pretrained_adaptiveST_1/checkpoint${cpk}.pth" \
            --output_dir "./OpenUS/eval_linear_output/vmamba_small_4gpu_pretrained_adaptiveST_busi_1/vmamba_small_lr${LR}_avg${AVGPOOL}_ckp${cpk}" \
            --log_name "vmamba_small_lr${LR}_avg${AVGPOOL}_ckp${cpk}" \
            --avgpool_patchtokens $AVGPOOL \
            --lr $LR \
            --pretrained_vmamba True \
            --dataset busi
        done
    done
done

echo "All folders processed!"
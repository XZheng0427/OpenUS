#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --mem=0 
#SBATCH --array=0-4
#SBATCH --job-name=validation_seg_vmamba_small_1_tn3k_from130_150_folder_%a
#SBATCH --output=./logs/slurm_validation_seg_vmamba_small_1_tn3k_from130_150_folder_%a_%j.out
#SBATCH --error=./logs/slurm_validation_seg_vmamba_small_1_tn3k_from130_150_folder_%a_%j.err

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
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Define parameter arrays
LR_VALUES=(0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)

cpk_values=(0130 0140 0150)

# Get the folder number from SLURM array task ID
folder=$SLURM_ARRAY_TASK_ID

echo "Processing folder $folder..."
for cpk in "${cpk_values[@]}"; do
    for LR in "${LR_VALUES[@]}"; do

        python eval_segmentation.py \
            --arch vmamba_small \
            --dataset_name 'TN3K' \
            --data_root './tn3k/trainval-image/' \
            --data_root2 './tn3k/trainval-mask/' \
            --json_file "./tn3k/tn3k-trainval-fold${folder}.json" \
            --pretrained_vmamba True \
            --pretrained_weights "./OpenUS/results/vmamba_small_4gpu_pretrained_adaptiveST_1/checkpoint_${cpk}.pth" \
            --output_dir "./OpenUS/eval_seg_output/vmamba_small_1_adaptiveST_1tn3k_folder${folder}_cpk${cpk}" \
            --log_name "vmamba_small_ir${LR}_mambaDecoder_folder${folder}_cpk${cpk}" \
            --lr $LR
    done
done

echo "Completed folder $folder"
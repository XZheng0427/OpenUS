#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4   
#SBATCH --gres=gpu:4
#SBATCH --partition=workq
#SBATCH --mem=0 
#SBATCH --job-name=train_openus_vmamba_small_4gpu
#SBATCH --output=./logs/train_openus_vmamba_small_4gpu_%j.out
#SBATCH --error=./logs/train_openus_vmamba_small_4gpu_%j.err

# Initialize conda and activate environment
source ~/miniforge3/bin/activate
module load cuda/12.2
conda activate vmamba-arm

PROJECT_DIR="./OpenUS"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
echo 'The work dir is: ' $PROJECT_DIR

# Set distributed training environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set WandB API key (replace YOUR_API_KEY with your actual key)
export WANDB_API_KEY="."

# pre-training with torchrun for distributed training
torchrun --nproc_per_node=4 \
        --master_addr=127.0.0.1 \
        --master_port=29501 \
        main_ibot_2.py \
    --arch vmamba_small \
    --patch_size 4 \
    --batch_size_per_gpu 32 \
    --num_workers 8 \
    --train_num 1 \
    --pretrained_vmamba True \
    --output_dir '.' \
    --global_crops_scale 0.14 1.0 \
    --mask_model 'atten_guided' \
    --masking_ratio 0.60 \
    --local_rec_loss True \
    --global_rec_loss True \
    --student_feedback True \
    --adaptive_weighting True \
    --alpha_init 0.1 \
    --alpha_final 0.9 \
    --alpha_schedule 'cosine' \
    --enable_wandb False \
    --wandb_name '' \
    --debug False  \
    --load_from '' \
    --epochs 151




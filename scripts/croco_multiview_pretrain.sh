#!/bin/bash
#SBATCH --job-name=croco_pretraining
#SBATCH --partition=gpu-a40
#SBATCH --account=account
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gpus=16
#SBATCH --mem=500G
#SBATCH --time=180:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user_email
# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate py38


python /gscratch/sciencehub/kmarathe/models/MIMIC/MIMIC/model/submitit_pretrain.py \
    --job_dir job_dir \
    --base_data_path base_data_path \
    --train_path_csv train/path/csv \
    --multiview \
    --accum_iter 4 \
    --nodes 2 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.9 \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1.5e-4 \
    --account account \
    --partition gpu-a40 \
    --timeout 20000 --report_to_wandb \
    --wandb_project croco --wandb_entity entity \
    --run_name croco_200_vitb \


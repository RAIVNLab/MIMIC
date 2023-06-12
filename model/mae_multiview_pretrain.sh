#!/bin/bash
#SBATCH --job-name=mae_pretraining
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gpus=16
#SBATCH --mem=500G
#SBATCH --time=180:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kmarathe@uw.edu
# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate py38


python /gscratch/sciencehub/kmarathe/models/SSL/MAE/mae/submitit_pretrain.py \
    --job_dir /gscratch/sciencehub/kmarathe/models/SSL/pt800_mae_multiview/ \
    --multiview \
    --accum_iter 4 \
    --nodes 2 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data/yfcc-tmp/imagenet/ \
    --train_path_csv /gscratch/sciencehub/kmarathe/models/SSL/visualize_views/Archive_1/train.csv

































# -m torch.distributed.launch --nproc_per_node=4
# --resume /gscratch/sciencehub/kmarathe/models/SSL/scripts_mae_pretraining/pt300_mae/checkpoint-160.pth --start_epoch 161 \
# python /gscratch/sciencehub/kmarathe/models/SSL/MAE/mae/main_pretrain.py \
#     --log_dir /gscratch/sciencehub/kmarathe/models/SSL/scripts/mae_job \
#     --nodes 1 \
#     --use_volta32 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path /data/yfcc-tmp/imagenet \
#    --partition gpu-a40
#   --account sciencehub
#SBATCH --account=sciencehub
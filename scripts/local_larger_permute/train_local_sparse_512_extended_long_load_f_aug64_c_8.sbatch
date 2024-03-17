#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_local
#SBATCH --output=train_local.out

eval "$(conda shell.bash hook)"
conda activate STGL

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train --val_positive_dist_threshold 512 --database_size 1536 --num_steps 300000 --two_stages --corr_level 4 --restore_ckpt logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-14_18-05-49-e1a998c0-c35f-4929-9e48-9b77c1c45242/STHN.pth --finetune --detach --augment_two_stages 64 --permute img --permute_max 8 --rotate_max 0.523599 --resize_max 0.3
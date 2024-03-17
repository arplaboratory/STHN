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
conda activate STHN

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train --database_size 1536 --two_stages --corr_level 4 --restore_ckpt logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-47-27-873eee1d-0406-4b02-bec3-088c7c03cc59/RHWF.pth --finetune --detach --augment_two_stages 64
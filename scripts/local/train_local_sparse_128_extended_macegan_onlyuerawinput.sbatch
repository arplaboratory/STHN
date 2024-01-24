#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --job-name=train_local
#SBATCH --output=train_local.out

eval "$(conda shell.bash hook)"
conda activate STGL

python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_sparse_128_exclusion --output IHN_results/satellite_thermal_ext_sparse_128_macegan --val_positive_dist_threshold 128 --use_ue --GAN_mode macegan --train_ue_method train_only_ue_raw_input
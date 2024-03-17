#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16               # uses 1 compute core per task
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --job-name=generate_extended
#SBATCH --output=generate_extended.out

eval "$(conda shell.bash hook)"
conda activate STHN

python3 eval_pix2pix_generate_h5_exclude.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_satellite_0_dense --datasets_folder ./datasets --G_net unet --GAN_upsample bilinear --GAN_resize 512 512
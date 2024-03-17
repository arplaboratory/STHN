#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=eval_local
#SBATCH --output=eval_local.out

eval "$(conda shell.bash hook)"
conda activate STGL

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/$MODEL/STHN.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --test --permute img --permute_max $PER --rotate_max $ROT --resize_max $RES
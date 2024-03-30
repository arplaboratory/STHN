#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=eval_local
#SBATCH --output=eval_local.out

eval "$(conda shell.bash hook)"
conda activate STGL

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/$MODEL/RHWF.pth --val_positive_dist_threshold $DC --lev0 --test --arch $ARCH
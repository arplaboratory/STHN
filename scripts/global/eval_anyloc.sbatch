#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=eval_thermal

eval "$(conda shell.bash hook)"
conda activate STHN

python3 global_pipeline/eval_anyloc.py --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --infer_batch_size 1 --prior_location_threshold=512 --resize 504 504
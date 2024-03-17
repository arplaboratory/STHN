#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=eval_thermal

eval "$(conda shell.bash hook)"
conda activate STHN

python3 global/eval.py --resume='logs/global_retrieval/'$1'/best_model.pth' --dataset_name=satellite_0_thermalmapping_135_dense --datasets_folder ./datasets --aggregation gem --infer_batch_size 16 --backbone $2 --fc_output_dim $3 --G_contrast $4

python3 global/eval.py --resume='logs/global_retrieval/'$1'/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --aggregation gem --infer_batch_size 16 --prior_location_threshold=512 --backbone $2 --fc_output_dim $3 --G_contrast $4
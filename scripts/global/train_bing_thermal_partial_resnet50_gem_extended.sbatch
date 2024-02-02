#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --job-name=train_sgm
#SBATCH --output=train_sgm.out

eval "$(conda shell.bash hook)"
conda activate STGL

python3 -u global_pipeline/train.py  --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion --backbone resnet50conv5 --aggregation gem --mining partial --datasets_folder ./datasets --save_dir global_retrieval --lr 0.00001 --fc_output_dim $1 --train_batch_size 16 --infer_batch_size 64 --num_workers 16 --epochs_num 40 --patience 40 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000 --unfreeze --use_extended_data --G_contrast manual
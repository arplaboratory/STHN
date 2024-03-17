#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --job-name=train_sgm
#SBATCH --output=train_sgm.out

eval "$(conda shell.bash hook)"
conda activate STHN

python3 -u global_pipeline/train.py  --dataset_name satellite_0_thermalmapping_135_contrast_dense_exclusion --backbone resnet50conv4 --aggregation netvlad --mining partial --datasets_folder ./datasets --save_dir global_retrieval --lr 0.00001 --fc_output_dim $FC --train_batch_size 16 --infer_batch_size 256 --num_workers 8 --epochs_num 100 --patience 50 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000 --unfreeze --use_extended_data --G_contrast manual
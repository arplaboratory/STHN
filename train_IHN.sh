#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STGL

python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense --output IHN_results/satellite_thermal_dense

python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_dense

python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_contrast_dense_exclusion --output IHN_results/satellite_thermal_ext_dense_contrast --G_contrast manual

python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_64 --val_positive_dist_threshold 64
python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_128 --val_positive_dist_threshold 128
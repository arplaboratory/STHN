#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STGL

# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense --output IHN_results/satellite_thermal_dense

# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_dense
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_dense_vanilla --use_ue --GAN_mode vanilla
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_dense_lsgan --use_ue --GAN_mode lsgan
python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_dense_macegan --use_ue --GAN_mode macegan --lr 1e-5

# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_contrast_exclusion --output IHN_results/satellite_thermal_ext_dense_contrast --G_contrast manual
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_contrast_exclusion --output IHN_results/satellite_thermal_ext_dense_contrast_vanilla --G_contrast manual --use_ue --GAN_mode vanilla
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_contrast_exclusion --output IHN_results/satellite_thermal_ext_dense_contrast_lsgan --G_contrast manual --use_ue --GAN_mode lsgan
python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_dense_contrast_exclusion --output IHN_results/satellite_thermal_ext_dense_contrast_macegan --G_contrast manual --use_ue --GAN_mode macegan --lr 1e-5

# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_64 --val_positive_dist_threshold 64
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_64_vanilla --val_positive_dist_threshold 64 --use_ue --GAN_mode vanilla
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_64_lsgan --val_positive_dist_threshold 64 --use_ue --GAN_mode lsgan
python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_64_macegan --val_positive_dist_threshold 64 --use_ue --GAN_mode macegan --lr 1e-5

# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_128 --val_positive_dist_threshold 128
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_128_vanilla --val_positive_dist_threshold 128 --use_ue --GAN_mode vanilla
# python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_128_lsgan --val_positive_dist_threshold 128 --use_ue --GAN_mode lsgan
python3 ./IHN/train_4cor.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --output IHN_results/satellite_thermal_ext_sparse_128_macegan --val_positive_dist_threshold 128 --use_ue --GAN_mode macegan --lr 1e-5
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STGL

python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --model IHN_results/satellite_thermal_dense/RHWF.pth --lev0

python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_dense/RHWF.pth --lev0

python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_contrast_dense_exclusion --model IHN_results/satellite_thermal_ext_dense_contrast/RHWF.pth --G_contrast manual --lev0

python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_sparse_64/RHWF.pth --val_positive_dist_threshold 64 --lev0
python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_sparse_128/RHWF.pth --val_positive_dist_threshold 128 --lev0
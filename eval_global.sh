#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STGL

# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --model IHN_results/satellite_thermal_dense_smalldb/RHWF.pth --lev0

# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --model IHN_results/satellite_thermal_ext_dense/RHWF.pth --lev0

# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --model IHN_results/satellite_thermal_ext_contrast_dense/RHWF.pth --G_contrast manual --lev0

# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_sparse_64/RHWF.pth --val_positive_dist_threshold 64 --lev0
# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_sparse_64_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_sparse_64_macegan/RHWF.pth --val_positive_dist_threshold 64 --lev0 --use_ue --GAN_mode macegan

# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_sparse_128/RHWF.pth --val_positive_dist_threshold 128 --lev0
# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_sparse_128_nocontrast_exclusion --model IHN_results/satellite_thermal_ext_sparse_128_macegan/RHWF.pth --val_positive_dist_threshold 128 --lev0 --use_ue --GAN_mode macegan

# python3 ./IHN/myevaluate.py --dataset_name satellite_0_thermalmapping_135_nocontrast_sparse_64_exclusion --model IHN_results/satellite_thermal_ext_sparse_64_macegan_onlyuerawinput/190000_RHWF.pth --lev0 --use_ue --use_raw_input
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STGL

# dense
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --eval_model logs/local_he/satellite_0_thermalmapping_135-2024-02-01_01-03-45-26fde83c-f3a9-4fcb-b213-76ab9e8f2002/RHWF.pth --lev0

# dense ext
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-01_01-03-21-d360a3d2-91aa-4520-b7f2-fd6882010801/RHWF.pth --lev0

# dense contrast
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --eval_model logs/local_he/satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-01_01-03-21-5d2ebc27-c771-4951-a54f-ce5c803ba79f/RHWF.pth --G_contrast manual --lev0

# sparse 64 ext
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-01_01-03-45-074825c7-978c-4d4e-8445-9224345ab162/RHWF.pth --val_positive_dist_threshold 64 --lev0

# sparse 128 ext
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-01_11-10-39-d5d1f02a-4b3c-432a-97f7-10d0b8ac5222/RHWF.pth --val_positive_dist_threshold 128 --lev0

# sparse 256 ext
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_dense --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-01-31_22-21-09-83daaa97-f4e0-48d1-9467-4e320eaa8a9f/RHWF.pth --val_positive_dist_threshold 256 --lev0
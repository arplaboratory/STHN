#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STGL

# # # dense ind
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --identity --test

# # # sparse 64 ind
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --val_positive_dist_threshold 64 --identity --test

# # # sparse 128 ind
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --val_positive_dist_threshold 128 --identity --test

# # # sparse 256 ind
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --val_positive_dist_threshold 256 --identity --test

# # # sparse 512 ind
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --val_positive_dist_threshold 512 --identity --test

# # # dense
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135-2024-02-18_23-51-22-b4a53c79-8317-4e75-8e83-a8fbfe8b4990/RHWF.pth --lev0 

# # # sparse 64
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135-2024-02-18_23-51-51-e745f931-7702-4e78-a667-55b7687d7f95/RHWF.pth --val_positive_dist_threshold 64 --lev0 --test

# # # sparse 128
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135-2024-02-18_23-51-51-12110539-444f-455c-b0ce-79a9dcdeef16/RHWF.pth --val_positive_dist_threshold 128 --lev0 --test

# # # sparse 256
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135-2024-02-18_23-51-51-7ec129ed-4406-47a8-af0c-1da45e0f68fa/RHWF.pth --val_positive_dist_threshold 256 --lev0 --test

# # # dense ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-22-169329a3-1c93-4b45-bc63-4727aa9be61a/RHWF.pth --lev0 --test

# # # sparse 64 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-51-1d65dc4d-96d0-4c74-abb6-9ae36175966f/RHWF.pth --val_positive_dist_threshold 64 --lev0 --test

# # # sparse 128 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-51-d88540dd-c6de-4b77-a1c4-4a42eea846f8/RHWF.pth --val_positive_dist_threshold 128 --lev0 --test

# # # sparse 256 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-51-76eff29e-df5e-4dea-9989-9418b5cf6737/RHWF.pth --val_positive_dist_threshold 256 --lev0 --test

# # # dense ext large
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_large_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_large_train-2024-02-18_20-38-38-1fe4bb94-3706-4706-9474-36f9678e1c7e/RHWF.pth --lev0 --database_size 1024 --corr_level 4 --test

# # # sparse 64 ext large
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_large_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_large_train-2024-02-18_20-39-41-7164cfc1-446d-4884-ae94-65f79a7e5338/RHWF.pth --val_positive_dist_threshold 64 --lev0 --database_size 1024 --corr_level 4 --test 

# # # sparse 128 ext large
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_large_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_large_train-2024-02-18_20-40-27-82fa3092-8494-4105-9194-8c5ebdbee381/RHWF.pth --val_positive_dist_threshold 128 --lev0 --database_size 1024 --corr_level 4 --test 

# # # sparse 256 ext large
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_large_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_large_train-2024-02-18_20-41-29-9d8be359-2b94-425d-849b-aa81b1f6efc8/RHWF.pth --val_positive_dist_threshold 256 --lev0 --database_size 1024 --corr_level 4 --test 

# # # sparse 512 ext large
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_large_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_large_train-2024-02-18_20-45-22-4fc12443-d88e-422f-a1f6-47ccfb3497c4/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1024 --corr_level 4 --test 

# # # dense ext larger
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-47-27-873eee1d-0406-4b02-bec3-088c7c03cc59/RHWF.pth --lev0 --database_size 1536 --corr_level 4 --test 

# # # sparse 64 ext larger
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-15-e2aa1243-510e-4d85-b8d8-5fad4cfb46f3/RHWF.pth --val_positive_dist_threshold 64 --lev0 --database_size 1536 --corr_level 4 --test 

# # # sparse 128 ext larger
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-42-ed74e93e-0c1a-4926-9d12-21aa8c257e12/RHWF.pth --val_positive_dist_threshold 128 --lev0 --database_size 1536 --corr_level 4 --test 

# # sparse 256 ext larger
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-cfb3bb2c-e987-4c17-bdc0-731bc776dcdd/RHWF.pth --val_positive_dist_threshold 256 --lev0 --database_size 1536 --corr_level 4 --test 

sparse 512 ext larger
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-97a33213-80a2-4f50-9d85-9ad04d7df728/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --test 

# # dense ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-44-363c4b6f-8c14-4fc5-bd0a-e7ff8b5e12e8/RHWF.pth --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 64 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-42-a8cbcd8e-7d23-40bc-8168-d66d052e340b/RHWF.pth --val_positive_dist_threshold 64 --lev0 --database_size 1536 --corr_level 4 --two_stages 

# # sparse 128 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-42-c3019a98-f4fa-43b1-aff4-281fd5e2ff8e/RHWF.pth --val_positive_dist_threshold 128 --lev0 --database_size 1536 --corr_level 4 --two_stages

# # sparse 256 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-38-e4623c02-945c-4605-8068-158cc52c2911/RHWF.pth --val_positive_dist_threshold 256 --lev0 --database_size 1536 --corr_level 4 --two_stages

# # sparse 512 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-43-328541ca-33bc-4c73-844b-3fef4b5a4ada/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages

# # sparse 512 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-6859fa15-4e14-44dc-95cb-d9ed28da6ae7/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages

# # dense ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-e1511359-bfe1-4f69-be80-3e36fe7c0443/RHWF.pth --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 64 ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-56-34-d29a4e76-5d9b-44da-a306-cda1de41a251/RHWF.pth --val_positive_dist_threshold 64 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 128 ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-fcea9216-14f1-420a-a1e1-95b7e39c73cf/RHWF.pth --val_positive_dist_threshold 128 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 256 ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-265573e5-6138-4813-a94b-dc7237ae500c/RHWF.pth --val_positive_dist_threshold 256 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# sparse 512 ext larger 2 - 1 
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-6859fa15-4e14-44dc-95cb-d9ed28da6ae7/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger 2 - 1 - p
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-22_00-14-15-31e473fa-74e8-4439-8176-d4ca78046663/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --fine_padding 128 --test
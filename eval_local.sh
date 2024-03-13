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

# # sparse 512
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135-2024-02-23_00-55-08-7a29553a-ca15-49ec-a6ff-4474d5d5143e/RHWF.pth --val_positive_dist_threshold 512 --lev0 --test

# # # dense ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-22-169329a3-1c93-4b45-bc63-4727aa9be61a/RHWF.pth --lev0 --test

# # # sparse 64 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-51-1d65dc4d-96d0-4c74-abb6-9ae36175966f/RHWF.pth --val_positive_dist_threshold 64 --lev0 --test

# # # sparse 128 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-51-d88540dd-c6de-4b77-a1c4-4a42eea846f8/RHWF.pth --val_positive_dist_threshold 128 --lev0 --test

# # # sparse 256 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-18_23-51-51-76eff29e-df5e-4dea-9989-9418b5cf6737/RHWF.pth --val_positive_dist_threshold 256 --lev0 --test

# # sparse 512 ext
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-23_01-02-42-521e7d5d-1d44-4a3a-b78a-3fbc6dc491c1/RHWF.pth --val_positive_dist_threshold 512 --lev0 --test

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

# # sparse 512 ext larger
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-97a33213-80a2-4f50-9d85-9ad04d7df728/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --test

# # dense ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-44-363c4b6f-8c14-4fc5-bd0a-e7ff8b5e12e8/RHWF.pth --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 64 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-42-a8cbcd8e-7d23-40bc-8168-d66d052e340b/RHWF.pth --val_positive_dist_threshold 64 --lev0 --database_size 1536 --corr_level 4 --two_stages 

# # sparse 128 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-42-c3019a98-f4fa-43b1-aff4-281fd5e2ff8e/RHWF.pth --val_positive_dist_threshold 128 --lev0 --database_size 1536 --corr_level 4 --two_stages

# # sparse 256 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-38-e4623c02-945c-4605-8068-158cc52c2911/RHWF.pth --val_positive_dist_threshold 256 --lev0 --database_size 1536 --corr_level 4 --two_stages

# # sparse 512 ext larger 2
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_23-56-43-328541ca-33bc-4c73-844b-3fef4b5a4ada/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test --vis_all

# # dense ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-e1511359-bfe1-4f69-be80-3e36fe7c0443/RHWF.pth --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 64 ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-56-34-d29a4e76-5d9b-44da-a306-cda1de41a251/RHWF.pth --val_positive_dist_threshold 64 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 128 ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-fcea9216-14f1-420a-a1e1-95b7e39c73cf/RHWF.pth --val_positive_dist_threshold 128 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 256 ext larger 2 - 1
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-265573e5-6138-4813-a94b-dc7237ae500c/RHWF.pth --val_positive_dist_threshold 256 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger 2 - 1 
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-20_21-35-02-6859fa15-4e14-44dc-95cb-d9ed28da6ae7/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger 2 - 1 - p
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-22_00-14-15-31e473fa-74e8-4439-8176-d4ca78046663/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --fine_padding 128 --test

# # sparse 512 ext larger finetune
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-22_14-42-07-0d94a284-8ad0-4f39-aa0e-e4e8081af873/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # # sparse 512 ext larger lam3
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-22_14-41-44-e31c4de7-50ba-4c7b-ab33-a6096498a934/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # # sparse 512 ext larger lam5
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-22_14-41-44-61fd214b-06d6-4377-b11b-af098e141185/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # # sparse 512 ext larger lam10
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-24_23-23-47-368903a9-f310-4b1a-9ba8-6b00d27ae4ff/240000_RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # # sparse 512 ext larger lam20
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-24_23-24-34-c4ccd4bb-df86-4f0b-a990-ec161f38c177/220000_RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --test --eval_model_fine logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion-2024-02-01_11-10-39-d5d1f02a-4b3c-432a-97f7-10d0b8ac5222/RHWF.pth

# # sparse 512 ext larger finetune
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-24_23-25-39-75b66309-2c05-41ed-a5c8-c2f11359584d/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger detach
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-51-55-22841214-b36a-4249-abf1-5f3937d6f288/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger detach 16
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-17-04-57ee61ab-d59b-4306-be70-09b933646703/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger detach 32
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-24-36-6d018daa-564f-4dd9-a65e-b82da322d28f/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger detach 64
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-39-37-22be38af-2b74-4e2f-adb9-c96465e5789d/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --fine_padding 64 --test

# # sparse 512 ext larger detach finetune
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-54-08-1cabfe69-7d0a-4190-aee8-e88953378f51/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger detach 16 finetune
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-03-04-df02d4cb-8d6d-4f11-936a-2ec2deed9c7c/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # sparse 512 ext larger detach 32 finetune
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-944f1dfb-0e67-4668-960d-eff422093f67/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# sparse 512 ext larger detach 64 finetune
# python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_larger_ori_train --eval_model logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-5afecf08-ec45-4b38-a80d-e6abd04e4a72/RHWF.pth --val_positive_dist_threshold 512 --lev0 --database_size 1536 --corr_level 4 --two_stages --test

# # TRAIN FROM SCRATCH
# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-51-55-22841214-b36a-4249-abf1-5f3937d6f288 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-51-55-22841214-b36a-4249-abf1-5f3937d6f288 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-51-55-22841214-b36a-4249-abf1-5f3937d6f288 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-51-55-22841214-b36a-4249-abf1-5f3937d6f288 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-17-04-57ee61ab-d59b-4306-be70-09b933646703 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-17-04-57ee61ab-d59b-4306-be70-09b933646703 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-17-04-57ee61ab-d59b-4306-be70-09b933646703 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-17-04-57ee61ab-d59b-4306-be70-09b933646703 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-24-36-6d018daa-564f-4dd9-a65e-b82da322d28f scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-24-36-6d018daa-564f-4dd9-a65e-b82da322d28f scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-24-36-6d018daa-564f-4dd9-a65e-b82da322d28f scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-24-36-6d018daa-564f-4dd9-a65e-b82da322d28f scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-39-37-22be38af-2b74-4e2f-adb9-c96465e5789d scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-39-37-22be38af-2b74-4e2f-adb9-c96465e5789d scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-39-37-22be38af-2b74-4e2f-adb9-c96465e5789d scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_13-39-37-22be38af-2b74-4e2f-adb9-c96465e5789d scripts/local_larger/eval_local_sparse_512_extended.sbatch

# FINETUNE
# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-54-08-1cabfe69-7d0a-4190-aee8-e88953378f51 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-54-08-1cabfe69-7d0a-4190-aee8-e88953378f51 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-54-08-1cabfe69-7d0a-4190-aee8-e88953378f51 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-27_22-54-08-1cabfe69-7d0a-4190-aee8-e88953378f51 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-03-04-df02d4cb-8d6d-4f11-936a-2ec2deed9c7c scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-03-04-df02d4cb-8d6d-4f11-936a-2ec2deed9c7c scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-03-04-df02d4cb-8d6d-4f11-936a-2ec2deed9c7c scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-03-04-df02d4cb-8d6d-4f11-936a-2ec2deed9c7c scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-944f1dfb-0e67-4668-960d-eff422093f67 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-944f1dfb-0e67-4668-960d-eff422093f67 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-944f1dfb-0e67-4668-960d-eff422093f67 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-944f1dfb-0e67-4668-960d-eff422093f67 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-5afecf08-ec45-4b38-a80d-e6abd04e4a72 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=16,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-5afecf08-ec45-4b38-a80d-e6abd04e4a72 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-5afecf08-ec45-4b38-a80d-e6abd04e4a72 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-28_22-02-52-5afecf08-ec45-4b38-a80d-e6abd04e4a72 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# # FREEZE
# # Center 32
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_22-54-41-59ba5c56-44a2-4bc7-9cda-9c171c18004d scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_22-54-41-59ba5c56-44a2-4bc7-9cda-9c171c18004d scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_22-54-41-59ba5c56-44a2-4bc7-9cda-9c171c18004d scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_22-54-41-59ba5c56-44a2-4bc7-9cda-9c171c18004d scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center pad 32 6th
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_23-50-23-db76b235-b145-4b77-91a0-17b9885559d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_23-50-23-db76b235-b145-4b77-91a0-17b9885559d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_23-50-23-db76b235-b145-4b77-91a0-17b9885559d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-03_23-50-23-db76b235-b145-4b77-91a0-17b9885559d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center 64 7th
# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-40-21-da9d6c25-7d02-4fe6-b65f-da70d45ef607 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-40-21-da9d6c25-7d02-4fe6-b65f-da70d45ef607 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-40-21-da9d6c25-7d02-4fe6-b65f-da70d45ef607 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-40-21-da9d6c25-7d02-4fe6-b65f-da70d45ef607 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center pad 64 5th
# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-43-22-36d5161c-8b19-4283-888a-822595895f05 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-43-22-36d5161c-8b19-4283-888a-822595895f05 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-43-22-36d5161c-8b19-4283-888a-822595895f05 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-43-22-36d5161c-8b19-4283-888a-822595895f05 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center 128
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-47-53-0edc92a4-87b7-49f4-ba94-ae98a51b2f13 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-47-53-0edc92a4-87b7-49f4-ba94-ae98a51b2f13 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-47-53-0edc92a4-87b7-49f4-ba94-ae98a51b2f13 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_02-47-53-0edc92a4-87b7-49f4-ba94-ae98a51b2f13 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center pad 128 3rd 
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_03-35-12-86a82f1e-9874-4942-8b74-6caed603a38b scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_03-35-12-86a82f1e-9874-4942-8b74-6caed603a38b scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_03-35-12-86a82f1e-9874-4942-8b74-6caed603a38b scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_03-35-12-86a82f1e-9874-4942-8b74-6caed603a38b scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # FINETUNE

# # Center 32 2nd 
# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-01-52-9e7ad31f-5f1a-4ab2-954f-3b56b5b6fa65 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-01-52-9e7ad31f-5f1a-4ab2-954f-3b56b5b6fa65 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-01-52-9e7ad31f-5f1a-4ab2-954f-3b56b5b6fa65 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-01-52-9e7ad31f-5f1a-4ab2-954f-3b56b5b6fa65 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center 64 1st
# sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-12-07-99176381-f32b-4207-b82d-2850d1f999c0 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-12-07-99176381-f32b-4207-b82d-2850d1f999c0 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-12-07-99176381-f32b-4207-b82d-2850d1f999c0 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-12-07-99176381-f32b-4207-b82d-2850d1f999c0 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center pad 64
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-24-48-c9983ad4-1351-443b-89b7-f96b26b6e319 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-24-48-c9983ad4-1351-443b-89b7-f96b26b6e319 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-24-48-c9983ad4-1351-443b-89b7-f96b26b6e319 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_04-24-48-c9983ad4-1351-443b-89b7-f96b26b6e319 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center 128
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_05-54-35-7689bf20-a071-488b-89ac-4b687f941a55 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_05-54-35-7689bf20-a071-488b-89ac-4b687f941a55 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_05-54-35-7689bf20-a071-488b-89ac-4b687f941a55 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_05-54-35-7689bf20-a071-488b-89ac-4b687f941a55 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# # Center pad 128
# # sbatch --export=ALL,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_06-02-00-50023460-e305-45ab-a877-f53648ba3464 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_06-02-00-50023460-e305-45ab-a877-f53648ba3464 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_06-02-00-50023460-e305-45ab-a877-f53648ba3464 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# # sbatch --export=ALL,PAD=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-04_06-02-00-50023460-e305-45ab-a877-f53648ba3464 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,DC=50,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-10_06-23-12-bb2e1255-f141-40d3-8878-2e87ba136df4 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=64,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-10_06-23-12-7a975cd7-008f-45d9-9ed2-0b4afd74827d scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=128,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-10_06-23-12-118e5a78-7844-4458-87a4-4a87c1eecad6 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=256,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-06_13-20-24-e680e844-6e49-4be7-895b-388f9292b6d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,DC=50,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-47-27-873eee1d-0406-4b02-bec3-088c7c03cc59 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=64,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-15-e2aa1243-510e-4d85-b8d8-5fad4cfb46f3 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=128,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-42-ed74e93e-0c1a-4926-9d12-21aa8c257e12 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=256,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-cfb3bb2c-e987-4c17-bdc0-731bc776dcdd scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=512,PAD=0,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-97a33213-80a2-4f50-9d85-9ad04d7df728 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,DC=512,PAD=64,IT0=6,IT1=3,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-06_17-02-30-1055cbb4-3afd-4ca5-b77c-c3b38a5d2fbc scripts/local_larger_2/eval_local_sparse_512_extended_it.sbatch
# sbatch --export=ALL,DC=512,PAD=64,IT0=3,IT1=3,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-06_13-20-24-e36e96c0-c69b-4e40-ac29-7eedfb8affe7 scripts/local_larger_2/eval_local_sparse_512_extended_it.sbatch
# sbatch --export=ALL,DC=512,PAD=64,IT0=3,IT1=6,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-06_13-22-27-b2791312-a2f6-45df-b106-43e259b3e833 scripts/local_larger_2/eval_local_sparse_512_extended_it.sbatch

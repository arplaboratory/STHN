#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STHN

# sbatch --export=ALL,DC=50,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-47-27-873eee1d-0406-4b02-bec3-088c7c03cc59 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=64,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-15-e2aa1243-510e-4d85-b8d8-5fad4cfb46f3 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=128,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-42-ed74e93e-0c1a-4926-9d12-21aa8c257e12 scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=256,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-cfb3bb2c-e987-4c17-bdc0-731bc776dcdd scripts/local_larger/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=512,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-97a33213-80a2-4f50-9d85-9ad04d7df728 scripts/local_larger/eval_local_sparse_512_extended.sbatch

# sbatch --export=ALL,DC=50,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-10_06-23-12-bb2e1255-f141-40d3-8878-2e87ba136df4 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=64,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-10_06-23-12-7a975cd7-008f-45d9-9ed2-0b4afd74827d scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=128,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-10_06-23-12-118e5a78-7844-4458-87a4-4a87c1eecad6 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=256,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-06_13-20-24-e680e844-6e49-4be7-895b-388f9292b6d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
# sbatch --export=ALL,DC=512,PAD=32,MODEL=satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-06_13-20-24-e680e844-6e49-4be7-895b-388f9292b6d8 scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
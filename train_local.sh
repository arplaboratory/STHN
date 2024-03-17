#!/bin/bash

## NORMAL
# sbatch ./scripts/local/train_local_dense_extended.sbatch
# sbatch ./scripts/local/train_local_dense.sbatch

# sbatch ./scripts/local/train_local_sparse_64_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_256_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_64.sbatch
# sbatch ./scripts/local/train_local_sparse_128.sbatch
# sbatch ./scripts/local/train_local_sparse_256.sbatch
# sbatch ./scripts/local/train_local_sparse_512.sbatch
# sbatch ./scripts/local/train_local_sparse_512_extended.sbatch

## LARGE
# sbatch ./scripts/local_large/train_local_dense_extended.sbatch
# sbatch ./scripts/local_large/train_local_sparse_64_extended.sbatch
# sbatch ./scripts/local_large/train_local_sparse_128_extended.sbatch
# sbatch ./scripts/local_large/train_local_sparse_256_extended_long.sbatch
# sbatch ./scripts/local_large/train_local_sparse_512_extended_long.sbatch

## LARGER
# sbatch scripts/local_larger/train_local_dense_extended.sbatch
# sbatch scripts/local_larger/train_local_sparse_64_extended.sbatch
# sbatch scripts/local_larger/train_local_sparse_128_extended.sbatch
# sbatch scripts/local_larger/train_local_sparse_256_extended_long.sbatch
# sbatch scripts/local_larger/train_local_sparse_512_extended_long.sbatch

## LARGER Permute
# sbatch scripts/local_larger_permute/train_local_sparse_512_extended_long.sbatch

## 2 stage
# sbatch scripts/local_larger_2/train_local_dense_extended_load_f_aug64_c.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_64_extended_load_f_aug64_c.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_128_extended_load_f_aug64_c.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long_load_f_aug64_c.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long_load_f_aug64_c.sbatch

## 2 stage permute
# sbatch scripts/local_larger_permute/train_local_sparse_512_extended_long_load_f_aug64_c.sbatch
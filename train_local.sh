#!/bin/bash

## NORMAL
# CE and EXT for DENSE
# sbatch ./scripts/local/train_local_dense_extended_ce.sbatch
# sbatch ./scripts/local/train_local_dense_extended.sbatch
# sbatch ./scripts/local/train_local_dense.sbatch

# EXT for SPARSE
# sbatch ./scripts/local/train_local_sparse_64_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_256_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_64.sbatch
# sbatch ./scripts/local/train_local_sparse_128.sbatch
# sbatch ./scripts/local/train_local_sparse_256.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_ce.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_ce.sbatch
# sbatch ./scripts/local/train_local_sparse_256_extended_ce.sbatch

# UE
# sbatch ./scripts/local/train_local_sparse_256_extended_macegan.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan.sbatch
# sbatch ./scripts/local/train_local_dense_extended_macegan.sbatch

# UE ONLY RAW INPUT
# sbatch ./scripts/local/train_local_dense_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_dense_extended_macegan_onlyuerawinput.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan_onlyuerawinput.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan_onlyuerawinput.sbatch
sbatch scripts/local/train_local_sparse_512_extended_ueri_64_permute.sbatch
sbatch scripts/local/train_local_sparse_512_extended_ueri_64.sbatch
sbatch scripts/local/train_local_sparse_512_extended_ueri_128_permute.sbatch
sbatch scripts/local/train_local_sparse_512_extended_ueri_128.sbatch
sbatch scripts/local/train_local_sparse_512_extended_ueri_256_permute.sbatch
sbatch scripts/local/train_local_sparse_512_extended_ueri_256.sbatch

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
# sbatch scripts/local_larger/train_local_sparse_512_extended_long_corr6.sbatch
# sbatch scripts/local_larger/train_local_sparse_512_extended_long_it12.sbatch
# sbatch scripts/local_larger/train_local_sparse_512_extended_long_iterative.sbatch
# sbatch scripts/local_larger/train_local_sparse_256_extended_long_lr.sbatch
# sbatch scripts/local_larger/train_local_sparse_512_extended_long_lr.sbatch

## 2 stage
# sbatch scripts/local_larger_2/train_local_dense_extended.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_64_extended.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_128_extended.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long_lam3.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long_lam5.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long_lam3.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long_lam5.sbatch

## 2 stage load prev
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long_load.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long_load.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_128_extended_load.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_64_extended_load.sbatch
# sbatch scripts/local_larger_2/train_local_dense_extended_load.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long_load_p.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long_load_p.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_128_extended_load_p.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_64_extended_load_p.sbatch
# sbatch scripts/local_larger_2/train_local_dense_extended_load_p.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_256_extended_long_load_f.sbatch
# sbatch scripts/local_larger_2/train_local_sparse_512_extended_long_load_f.sbatch

# larger permute
# sbatch scripts/local_larger_permute/train_local_sparse_512_extended_long.sbatch
# sbatch scripts/local_larger_permute/train_local_sparse_256_extended_long.sbatch
# sbatch scripts/local_larger_permute/train_local_sparse_128_extended.sbatch
# sbatch scripts/local_larger_permute/train_local_sparse_64_extended.sbatch
# sbatch scripts/local_larger_permute/train_local_dense_extended.sbatch
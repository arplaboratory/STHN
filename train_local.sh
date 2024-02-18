#!/bin/bash

# CE and EXT for DENSE
sbatch ./scripts/local/train_local_dense_extended_contrast.sbatch
sbatch ./scripts/local/train_local_dense_contrast.sbatch
sbatch ./scripts/local/train_local_dense_extended.sbatch
sbatch ./scripts/local/train_local_dense.sbatch

# EXT for SPARSE
sbatch ./scripts/local/train_local_sparse_64_extended.sbatch
sbatch ./scripts/local/train_local_sparse_128_extended.sbatch
sbatch ./scripts/local/train_local_sparse_256_extended.sbatch
sbatch ./scripts/local/train_local_sparse_64.sbatch
sbatch ./scripts/local/train_local_sparse_128.sbatch
sbatch ./scripts/local/train_local_sparse_256.sbatch

# UE
# sbatch ./scripts/local/train_local_dense_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_dense_extended_macegan_onlyuerawinput.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan_onlyuerawinput.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan_onlyuerawinput.sbatch


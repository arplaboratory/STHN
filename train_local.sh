#!/bin/bash

sbatch ./scripts/local/train_local_dense_extended_contrast.sbatch
# sbatch ./scripts/local/train_local_dense_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_dense_extended_macegan_onlyuerawinput.sbatch
sbatch ./scripts/local/train_local_dense_extended.sbatch
sbatch ./scripts/local/train_local_dense.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_sparse_64_extended_macegan_onlyuerawinput.sbatch
sbatch ./scripts/local/train_local_sparse_64_extended.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan_onlyue.sbatch
# sbatch ./scripts/local/train_local_sparse_128_extended_macegan_onlyuerawinput.sbatch
sbatch ./scripts/local/train_local_sparse_128_extended.sbatch
sbatch ./scripts/local/train_local_sparse_256_extended.sbatch
#!/bin/bash

sbatch scripts/local/train_local_dense_extended_lsgan.sbatch
sbatch scripts/local/train_local_dense_extended_macegan.sbatch
sbatch scripts/local/train_local_dense_extended.sbatch
sbatch scripts/local/train_local_dense_smalldb.sbatch
sbatch scripts/local/train_local_dense.sbatch
sbatch scripts/local/train_local_sparse_64_extended_macegan.sbatch
sbatch scripts/local/train_local_sparse_64_extended.sbatch
sbatch scripts/local/train_local_sparse_128_extended_macegan.sbatch
sbatch scripts/local/train_local_sparse_128_extended.sbatch
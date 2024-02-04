#!/bin/bash

sbatch --export=ALL,FC=2048 scripts/global/train_bing_thermal_partial_resnet50_gem_extended_sparse_64.sbatch 
sbatch --export=ALL,FC=2048 scripts/global/train_bing_thermal_partial_resnet50_gem_extended_sparse_128.sbatch
sbatch --export=ALL,FC=2048 scripts/global/train_bing_thermal_partial_resnet50_gem_extended_sparse_256.sbatch
sbatch --export=ALL,FC=2048 scripts/global/train_bing_thermal_partial_resnet50_gem_extended.sbatch
sbatch --export=ALL,FC=2048 scripts/global/train_bing_thermal_partial_resnet50_gem_nocontrast_extended.sbatch
sbatch --export=ALL,FC=4096 scripts/global/train_bing_thermal_partial_resnet50_gem_nocontrast.sbatch
sbatch --export=ALL,FC=2048 scripts/global/train_bing_thermal_partial_resnet50_gem_nocontrast.sbatch
sbatch --export=ALL,FC=1024 scripts/global/train_bing_thermal_partial_resnet50_gem_nocontrast.sbatch
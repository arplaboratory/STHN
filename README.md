# STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery

This is the official repository for [STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery](https://arxiv.org/abs/2306.02994).

```
bibtex TBD
```
**Developer: Jiuhong Xiao<br />
Affiliation: [NYU ARPL](https://wp.nyu.edu/arpl/)<br />
Maintainer: Jiuhong Xiao (jx1190@nyu.edu)<br />**

## Dataset
We extend the Boson-nighttime dataset from [STGL](https://github.com/arplaboratory/satellite-thermal-geo-localization/tree/main) with additional unpaired satellite images and our generated thermal images using TGM.

Dataset link (957 GB): [Download](https://drive.google.com/drive/folders/1HRUlR-X9u3VfWtSwh19DsvPIYU5Q3TPG?usp=sharing)

The ``datasets`` folder should be created in the root folder with the following structure. By default, the dataset uses $W_S=512$, while the ``larger`` suffix indicates $W_S=1536$.

```
STHN/datasets/
├── satellite_0_satellite_0_dense
│   └── train_database.h5
├── satellite_0_satellite_0_dense_larger_ori
│   └── train_database.h5
├── satellite_0_thermalmapping_135
│   ├── test_database.h5
│   ├── test_queries.h5
│   ├── train_database.h5
│   ├── train_queries.h5
│   ├── val_database.h5
│   └── val_queries.h5
├── satellite_0_thermalmapping_135_larger_ori
│   ├── test_database.h5
│   ├── train_database.h5
│   └── val_database.h5
├── satellite_0_thermalmapping_135_larger_ori_train
│   ├── test_database.h5 -> ../satellite_0_thermalmapping_135_larger_ori/test_database.h5
│   ├── test_queries.h5 -> ../satellite_0_thermalmapping_135/test_queries.h5
│   ├── train_database.h5 -> ../satellite_0_thermalmapping_135_larger_ori/train_database.h5
│   ├── train_queries.h5 -> ../satellite_0_thermalmapping_135/train_queries.h5
│   ├── val_database.h5 -> ../satellite_0_thermalmapping_135_larger_ori/val_database.h5
│   └── val_queries.h5 -> ../satellite_0_thermalmapping_135/val_queries.h5
├── satellite_0_thermalmapping_135_nocontrast_dense_exclusion
│   ├── extended_database.h5 -> ../satellite_0_satellite_0_dense/train_database.h5
│   ├── extended_queries.h5
│   ├── test_database.h5 -> ../satellite_0_thermalmapping_135/test_database.h5
│   ├── test_queries.h5 -> ../satellite_0_thermalmapping_135/test_queries.h5
│   ├── train_database.h5 -> ../satellite_0_thermalmapping_135/train_database.h5
│   ├── train_queries.h5 -> ../satellite_0_thermalmapping_135/train_queries.h5
│   ├── val_database.h5 -> ../satellite_0_thermalmapping_135/val_database.h5
│   └── val_queries.h5 -> ../satellite_0_thermalmapping_135/val_queries.h5
├── satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train
│   ├── extended_database.h5 -> ../satellite_0_satellite_0_dense_larger_ori/train_database.h5
│   ├── extended_queries.h5 -> ../satellite_0_thermalmapping_135_nocontrast_dense_exclusion/extended_queries.h5
│   ├── test_database.h5 -> ../satellite_0_thermalmapping_135_larger_ori/test_database.h5
│   ├── test_queries.h5 -> ../satellite_0_thermalmapping_135/test_queries.h5
│   ├── train_database.h5 -> ../satellite_0_thermalmapping_135_larger_ori/train_database.h5
│   ├── train_queries.h5 -> ../satellite_0_thermalmapping_135/train_queries.h5
│   ├── val_database.h5 -> ../satellite_0_thermalmapping_135_larger_ori/val_database.h5
│   └── val_queries.h5 -> ../satellite_0_thermalmapping_135/val_queries.h5
```

## Conda Environment Setup
Our repository requires a conda environment. Relevant packages are listed in ``env.yml``. Run the following command to setup the conda environment.
```
conda env create -f env.yml
```

## Training
You can find the training scripts and evaluation scripts in ``scripts`` folder. The scripts is for slurm system to submit sbatch job. If you want to run bash command, change the suffix from ``sbatch`` to ``sh`` and run with bash.

### Coarse-level Alignment Training

To train the coarse-level alignment module, use one of the scripts in ``./scripts/local`` for $W_S=512$ and ``./scripts/local_larger`` for $W_S=1536$ with different $D_C, for example:

```
./scripts/local_larger/train_local_sparse_512_extended_long.sbatch
```

After training, find your model folder in ``./logs/local_he/$dataset_name-$datetime-$uuid``  
The ``$dataset_name-$datetime-$uuid`` is your **coarse_model_folder_name**.



### Refinement Training (only for $W_S=1536)

Before training, change the ``restore_ckpt`` argument using **coarse_model_folder_name** to load your trained coarse-level alignment module.

To train the refinement module, use one of the scripts in ``./scripts/local_larger_2`` per name, for example:

```
./scripts/local/train_local_sparse_512_extended_long_load_f_aug64_c.sbatch
```

After training SGM, find your model folder in ``./logs/local_he/$dataset_name-$datetime-$uuid``  
The ``$dataset_name-$datetime-$uuid`` is your **refine_model_folder_name**.

## Evaluation
To evaluate one-stage and two-stage methods, use one of the following scripts:
```
./scripts/local/eval.sbatch
./scripts/local_larger/eval.sbatch
./scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
```

Find the test results in ``./test/local_he/$model_folder_name/``.

## Image-matching Baselines
For training and evaluate the image-matching baselines (anyloc and STGL), please refer to ``scripts/global/``.

## Acknowledgement
Our implementation refers to the following repositories and appreciate their excellent work.

https://github.com/imdumpl78/IHN  
https://github.com/AnyLoc/AnyLoc  
https://github.com/gmberton/deep-visual-geo-localization-benchmark  
https://github.com/fungtion/DANN 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

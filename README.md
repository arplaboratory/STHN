# STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery

This is the official repository for STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery.

```
@article{xiao2024sthn,
  title={STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery},
  author={Xiao, Jiuhong and Zhang, Ning and Tortei, Daniel and Loianno, Giuseppe},
  journal={arXiv preprint arXiv:2405.20470},
  year={2024}
}
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

To train the coarse-level alignment module, use one of the scripts in ``./scripts/local`` for $W_S=512$ and ``./scripts/local_larger`` for $W_S=1536$ with different $D_C$, for example:

```
./scripts/local_larger/train_local_sparse_512_extended_long.sbatch
```

After training, find your model folder in ``./logs/local_he/$dataset_name-$datetime-$uuid``  
The ``$dataset_name-$datetime-$uuid`` is your **coarse_model_folder_name**.



### Refinement Training (only for $W_S=1536$)

Before training, change the ``restore_ckpt`` argument using **coarse_model_folder_name** to load your trained coarse-level alignment module.

To train the refinement module, use one of the scripts in ``./scripts/local_larger_2`` per name, for example:

```
./scripts/local/train_local_sparse_512_extended_long_load_f_aug64_c.sbatch
```

After training, find your model folder in ``./logs/local_he/$dataset_name-$datetime-$uuid``  
The ``$dataset_name-$datetime-$uuid`` is your **refine_model_folder_name**.

## Evaluation
To evaluate one-stage and two-stage methods, use one of the following scripts:
```
./scripts/local/eval.sbatch
./scripts/local_larger/eval.sbatch
./scripts/local_larger_2/eval_local_sparse_512_extended.sbatch
```

Find the test results in ``./test/local_he/$model_folder_name/``.  
**Please note that MACE and CE tests are conducted on resized images (256x256). To convert these metrics into meters, multiply them by a scaling factor, denoted as $\alpha$. Specifically, set $\alpha$ to 6 when $W_S=1536$ and 2 when $W_S=512$.**

## Image-matching Baselines
For training and evaluating the image-matching baselines (anyloc and STGL), please refer to ``scripts/global/`` for training and evaluation.

## Pretrained Models
Download pretrained models for $W_S=1536$ and $D_C=512$ m: [Download](https://drive.google.com/drive/folders/19Z0wqeDlJyzTZB1qc355G2WEww7I9rTB?usp=sharing)

## Additional Details
<details>
  <summary>Train/Val/Test split</summary>
  Below is the visualization of the train-validation-test regions. The dataset includes thermal maps from six flights: three flights (conducted at 9 PM, 12 AM, and 2 AM) cover the upper region, and the other three flights (conducted at 10 PM, 1 AM, and 3 AM) cover the lower region. The lower region is further divided into training and validation subsets. The synthesized thermal images span a larger area (23,744m x 9,088m) but exclude the test region to assess generalization performance properly.
  
  ![image](https://github.com/arplaboratory/STHN/assets/29690116/8e833ba9-644e-4446-b951-7b17a5e4316b)
  
</details>
<details>
  <summary>Architecture Details</summary>
  The feature extractor consists of multiple residual blocks with multi-layer CNN and group normalization:  
  https://github.com/arplaboratory/STHN/blob/0ad04d7fb19ba369d24184cda80941640c618631/local_pipeline/extractor.py#L177
  The iterative updater is a multi-layer CNN with group normalization:  
  https://github.com/arplaboratory/STHN/blob/eed553fb45756ce5ea35418db77383732c444c42/local_pipeline/update.py#L299  
  The TGM is using the Pix2Pix paradigm:
  https://github.com/arplaboratory/STHN/blob/eed553fb45756ce5ea35418db77383732c444c42/global_pipeline/model/network.py#L273
  
</details>

<details>
  <summary>Direct Linear Transformation Details</summary>
  The Direct Linear Transformation (DLT) is used to solve the homography transformation matrix (3x3) given four corresponding point pairs.   
  
  In practice, we use kornia's implementation:  
  https://kornia.readthedocs.io/en/stable/geometry.transform.html#kornia.geometry.transform.get_perspective_transform   
  For more details of formulas, you can refer to: https://en.wikipedia.org/wiki/Direct_linear_transformation.
  
</details>

## Acknowledgement
Our implementation refers to the following repositories and appreciate their excellent work.

https://github.com/imdumpl78/IHN  
https://github.com/AnyLoc/AnyLoc  
https://github.com/gmberton/deep-visual-geo-localization-benchmark  
https://github.com/fungtion/DANN  
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

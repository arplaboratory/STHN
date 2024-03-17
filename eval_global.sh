#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate STHN

# # netvlad DANN
# python3 global_pipeline/eval.py --resume='logs/global_retrieval/satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-19_12-10-07-dd8b1b8b-d529-4277-b96c-2480b813eb69/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --aggregation netvlad --infer_batch_size 16 --prior_location_threshold=512 --backbone resnet50conv4 --fc_output_dim 4096 --G_contrast manual

# # netvlad
# python3 global_pipeline/eval.py --resume='logs/global_retrieval/satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-19_12-37-10-32caa09a-06c0-4549-a30e-f1e99424ed16/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --aggregation netvlad --infer_batch_size 16 --prior_location_threshold=512 --backbone resnet50conv4 --fc_output_dim 4096 --G_contrast manual

# # gem DANN
# python3 global_pipeline/eval.py --resume='logs/global_retrieval/satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-14_23-02-31-91400d55-5881-48e5-b6cb-cecff4f47a3f/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --aggregation gem --infer_batch_size 16 --prior_location_threshold=512 --backbone resnet50conv4 --fc_output_dim 4096 --G_contrast manual

# # gem
# python3 global_pipeline/eval.py --resume='logs/global_retrieval/satellite_0_thermalmapping_135_contrast_dense_exclusion-2024-02-14_23-05-05-be2c36a5-1841-4667-a95d-05d7cc0a7472/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --aggregation gem --infer_batch_size 16 --prior_location_threshold=512 --backbone resnet50conv4 --fc_output_dim 4096 --G_contrast manual
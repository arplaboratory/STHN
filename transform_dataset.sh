#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate STHN

# bing + bing
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 0 --compress --sample_method stride --region_num 1 --crop_width 1024 --stride 35 --generate_data database --maintain_size &

# # # # # bing + thermal_1
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 0 --compress --sample_method stride --region_num 1 --crop_width 1024 --generate_data database &

# # # # # bing + thermal_2
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 1 --compress --sample_method stride --region_num 2 --crop_width 1024 --generate_data database &

# # # # # bing + thermal_3
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 2 --compress --sample_method stride --region_num 1 --crop_width 1024 --generate_data database &

# # # # # bing + thermal_4
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 3 --compress --sample_method stride --region_num 2 --crop_width 1024 --generate_data database &

# # # # # bing + thermal_5
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 4 --compress --sample_method stride --region_num 1 --crop_width 1024 --generate_data database &

# # # # # bing + thermal_6
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 5 --compress --sample_method stride --region_num 2 --crop_width 1024 --generate_data database &

# bing + thermal_123456
# python global_pipeline/h5_merger.py --database_name satellite --database_indexes 0 --queries_name thermalmapping --queries_indexes 135 --compress --region_num 2 --generate_data database --resize_width 1536

# python global_pipeline/h5_merger.py --database_name satellite --database_indexes 0 --queries_name thermalmapping --queries_indexes 024 --compress --region_num 1 --generate_data database --resize_width 1536

# rm -r ./datasets/satellite_0_thermalmapping_0
# rm -r ./datasets/satellite_0_thermalmapping_1
# rm -r ./datasets/satellite_0_thermalmapping_2
# rm -r ./datasets/satellite_0_thermalmapping_3
# rm -r ./datasets/satellite_0_thermalmapping_4
# rm -r ./datasets/satellite_0_thermalmapping_5
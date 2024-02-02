#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate STGL

# bing + bing
python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 0 --compress --sample_method stride --region_num 1 --crop_width 1024 --stride 35

# # bing + bing
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 0 --compress --sample_method stride --region_num 1 --crop_width 1024 --stride 64

# # bing + bing
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 0 --compress --sample_method stride --region_num 1 --crop_width 1024 --stride 128

# # bing + bing
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 0 --compress --sample_method stride --region_num 1 --crop_width 1024 --stride 256

# # bing + foxtech
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name foxtechmapping --queries_index 0 --compress --sample_method stride --region_num 2 &

# # bing + esri
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 1 --compress --sample_method stride --region_num 3 &

# # bing + google
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name satellite --queries_index 3 --compress --sample_method stride --region_num 3 &

# # # bing + thermal_1
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 0 --compress --sample_method stride --region_num 1 &

# # # # bing + thermal_2
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 1 --compress --sample_method stride --region_num 2

# # # # bing + thermal_3
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 2 --compress --sample_method stride --region_num 1 &

# # # # bing + thermal_4
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 3 --compress --sample_method stride --region_num 2

# # # # bing + thermal_5
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 4 --compress --sample_method stride --region_num 1 &

# # # # bing + thermal_6
# python global_pipeline/h5_transformer.py --database_name satellite --database_index 0 --queries_name thermalmapping --queries_index 5 --compress --sample_method stride --region_num 2

# bing + ADASI
# python global_pipeline/h5_transformer.py --database_name ADASI --database_index 0 --queries_name ADASI_thermal --queries_index 0 --compress --sample_method stride --region_num 1 &

# bing + ADAIS_thermal
# python global_pipeline/h5_transformer.py --database_name ADASI --database_index 0 --queries_name ADASI_thermal --queries_index 0 --compress --sample_method stride --region_num 1 &

# bing + thermal_123456
# python h5_merger.py --database_name satellite --database_indexes 0 --queries_name thermalmapping --queries_indexes 135 --compress --region_num 2

# rm -r ./datasets/satellite_0_thermalmapping_0
# rm -r ./datasets/satellite_0_thermalmapping_1
# rm -r ./datasets/satellite_0_thermalmapping_2
# rm -r ./datasets/satellite_0_thermalmapping_3
# rm -r ./datasets/satellite_0_thermalmapping_4
# rm -r ./datasets/satellite_0_thermalmapping_5
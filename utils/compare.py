import h5py
from tqdm import tqdm

f1 = h5py.File("datasets/satellite_0_thermalmapping_135/train_queries.h5", "r")
f2 = h5py.File("datasets/satellite_0_thermalmapping_135_dense/train_queries.h5", "r")
# # compare image name
# for i in tqdm(range(len(f1["image_name"]))):
#     if f1["image_name"][i] != f2["image_name"][i]:
#         raise KeyError()
# compare image data
for i in tqdm(range(len(f1["image_data"]))):
    if (f1["image_data"][i] != f2["image_data"][i]).any():
        raise KeyError()


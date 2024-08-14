# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import kornia.geometry.transform as tgm

import random
from glob import glob
import os
import cv2
from os.path import join
import h5py
from sklearn.neighbors import NearestNeighbors
import logging
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import math
Image.MAX_IMAGE_PIXELS = None
marginal = 0
# patch_size = 256

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

TB_val_region = [2650, 5650, 5100, 9500]

inv_base_transforms = transforms.Compose(
    [ 
        # transforms.Normalize(mean = [ -m/s for m, s in zip(imagenet_mean, imagenet_std)],
        #                      std = [ 1/s for s in imagenet_std]),
        transforms.ToPILImage(),
    ]
)

class homo_dataset(data.Dataset):
    def __init__(self, args, augment=False):

        self.args = args
        self.is_test = False
        self.image_list_img1 = []
        self.image_list_img2 = []
        self.dataset=[]
        self.augment = augment
        if self.augment: # EVAL
            if self.args.eval_model is not None:
                self.rng = None
            self.augment_type = ["no"]
            if self.args.perspective_max > 0:
                self.augment_type.append("perspective")
            if self.args.rotate_max > 0:
                self.augment_type.append("rotate")
            if self.args.resize_max > 0:
                self.augment_type.append("resize")
        self.base_transform = transforms.Compose(
            [
                transforms.Resize([self.args.resize_width, self.args.resize_width]),
            ]
        )
        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ]
        )
        
    def rotate_transform(self, rotation, four_point_org, four_point_1, four_point_org_augment, four_point_1_augment):
        center_x_org = torch.tensor((self.args.resize_width - 1)/2)
        center_x_1 = (four_point_1[0, 0, :] + four_point_1[0, 3, :])/2
        four_point_org_augment[0, 0, 0] = (four_point_org[0, 0, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 0, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 0, 1] = (four_point_org[0, 0, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 0, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        four_point_org_augment[0, 1, 0] = (four_point_org[0, 1, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 1, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 1, 1] = (four_point_org[0, 1, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 1, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        four_point_org_augment[0, 2, 0] = (four_point_org[0, 2, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 2, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 2, 1] = (four_point_org[0, 2, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 2, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        four_point_org_augment[0, 3, 0] = (four_point_org[0, 3, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 3, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 3, 1] = (four_point_org[0, 3, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 3, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        four_point_1_augment[0, 0, 0] = (four_point_1[0, 0, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 0, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 0, 1] = (four_point_1[0, 0, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 0, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        four_point_1_augment[0, 1, 0] = (four_point_1[0, 1, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 1, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 1, 1] = (four_point_1[0, 1, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 1, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        four_point_1_augment[0, 2, 0] = (four_point_1[0, 2, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 2, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 2, 1] = (four_point_1[0, 2, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 2, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        four_point_1_augment[0, 3, 0] = (four_point_1[0, 3, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 3, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 3, 1] = (four_point_1[0, 3, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 3, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        # print("ori:", four_point_org[0, 0, 0], four_point_org[0, 0, 1], four_point_1[0, 0, 0], four_point_1[0, 0, 1])
        # print("now:", four_point_org_augment[0, 0, 0], four_point_org_augment[0, 0, 1], four_point_1_augment[0, 0, 0], four_point_1_augment[0, 0, 1])
        # print("center:", center_x_1, four_point_1[0, 0, :], four_point_1[0, 3, :])
        return four_point_org_augment, four_point_1_augment

    def resize_transform(self, scale_factor, beta, alpha, four_point_org_augment, four_point_1_augment):
        offset = self.args.resize_width * (1 - scale_factor) / 2
        four_point_org_augment[0, 0, 0] += offset
        four_point_org_augment[0, 0, 1] += offset
        four_point_org_augment[0, 1, 0] -= offset
        four_point_org_augment[0, 1, 1] += offset
        four_point_org_augment[0, 2, 0] += offset
        four_point_org_augment[0, 2, 1] -= offset
        four_point_org_augment[0, 3, 0] -= offset
        four_point_org_augment[0, 3, 1] -= offset
        four_point_1_augment[0, 0, 0] += offset * beta / alpha
        four_point_1_augment[0, 0, 1] += offset * beta / alpha
        four_point_1_augment[0, 1, 0] -= offset * beta / alpha
        four_point_1_augment[0, 1, 1] += offset * beta / alpha
        four_point_1_augment[0, 2, 0] += offset * beta / alpha
        four_point_1_augment[0, 2, 1] -= offset * beta / alpha
        four_point_1_augment[0, 3, 0] -= offset * beta / alpha
        four_point_1_augment[0, 3, 1] -= offset * beta / alpha
        return four_point_org_augment, four_point_1_augment

    def __getitem__(self, query_PIL_image, database_PIL_image, query_utm, database_utm, index, pos_index, neg_img2=None):
        if hasattr(self, "rng") and self.rng is None:
            worker_info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(seed=worker_info.id)

        img1 = query_PIL_image
        img2 = database_PIL_image # img1 warp to img2

        height, width = img1.size
        t = np.float32(np.array(query_utm - database_utm))
        t[0][0], t[0][1] = t[0][1], t[0][0] # Swap!
        
        # img1, img2, img2_ori = self.query_transform(img1), self.database_transform(img2), self.database_transform_ori(img2)
        img1 = self.query_transform(img1)
        alpha = self.args.database_size / self.args.resize_width
        t = t / alpha # align with the resized image
        
        t_tensor = torch.Tensor(t).squeeze(0)
        y_grid, x_grid = np.mgrid[0:self.args.resize_width, 0:self.args.resize_width]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()
        four_point_org = torch.zeros((2, 2, 2))
        top_left = torch.Tensor([0, 0])
        top_right = torch.Tensor([self.args.resize_width - 1, 0])
        bottom_left = torch.Tensor([0, self.args.resize_width - 1])
        bottom_right = torch.Tensor([self.args.resize_width - 1, self.args.resize_width - 1])
        four_point_org[:, 0, 0] = top_left
        four_point_org[:, 0, 1] = top_right
        four_point_org[:, 1, 0] = bottom_left
        four_point_org[:, 1, 1] = bottom_right
        four_point_1 = torch.zeros((2, 2, 2))
        if self.args.database_size == 512:
            four_point_1[:, 0, 0] = t_tensor + top_left
            four_point_1[:, 0, 1] = t_tensor + top_right
            four_point_1[:, 1, 0] = t_tensor + bottom_left
            four_point_1[:, 1, 1] = t_tensor + bottom_right
        elif self.args.database_size == 1024:
            top_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width/4])
            top_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width/4])
            bottom_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width - self.args.resize_width/4 - 1])
            bottom_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width - self.args.resize_width/4 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize
            four_point_1[:, 0, 1] = t_tensor + top_right_resize
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize
        elif self.args.database_size == 1536:
            top_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width/3])
            top_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width/3])
            bottom_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width - self.args.resize_width/3 - 1])
            bottom_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width - self.args.resize_width/3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize2
            four_point_1[:, 0, 1] = t_tensor + top_right_resize2
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize2
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize2
        elif self.args.database_size == 2048:
            top_left_resize3 = torch.Tensor([self.args.resize_width/8*3, self.args.resize_width/8*3])
            top_right_resize3 = torch.Tensor([self.args.resize_width - self.args.resize_width/8*3 - 1, self.args.resize_width/8*3])
            bottom_left_resize3 = torch.Tensor([self.args.resize_width/8*3, self.args.resize_width - self.args.resize_width/8*3 - 1])
            bottom_right_resize3 = torch.Tensor([self.args.resize_width - self.args.resize_width/8*3 - 1, self.args.resize_width - self.args.resize_width/8*3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize3
            four_point_1[:, 0, 1] = t_tensor + top_right_resize3
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize3
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize3
        elif self.args.database_size == 2560:
            top_left_resize4 = torch.Tensor([self.args.resize_width/5*2, self.args.resize_width/5*2])
            top_right_resize4 = torch.Tensor([self.args.resize_width - self.args.resize_width/5*2 - 1, self.args.resize_width/5*2])
            bottom_left_resize4 = torch.Tensor([self.args.resize_width/5*2, self.args.resize_width - self.args.resize_width/5*2 - 1])
            bottom_right_resize4 = torch.Tensor([self.args.resize_width - self.args.resize_width/5*2 - 1, self.args.resize_width - self.args.resize_width/5*2 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize4
            four_point_1[:, 0, 1] = t_tensor + top_right_resize4
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize4
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize4
        else:
            raise NotImplementedError()
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
        four_point_1 = four_point_1.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
        
        if self.augment:
            #augment
            #All point are in resized scales (256)
            four_point_org_augment = four_point_org.clone()
            four_point_1_augment = four_point_1.clone()
            beta = self.args.crop_width/self.args.resize_width
            if self.args.eval_model is None or self.args.multi_aug_eval: # EVAL
                augment_type_single = random.choice(self.augment_type)
                if augment_type_single == "rotate":
                    rotation = torch.tensor(random.random() - 0.5) * 2 * self.args.rotate_max # on 256x256
                    four_point_org_augment, four_point_1_augment = self.rotate_transform(rotation, four_point_org, four_point_1, four_point_org_augment, four_point_1_augment)
                elif augment_type_single == "resize":
                    scale_factor = 1 + (random.random() - 0.5) * 2 * self.args.resize_max # on 256x256
                    assert scale_factor > 0
                    four_point_org_augment, four_point_1_augment = self.resize_transform(scale_factor, beta, alpha, four_point_org_augment, four_point_1_augment)
                elif augment_type_single == "perspective":
                    for p in range(4):
                        for xy in range(2):
                            t1 = random.randint(-self.args.perspective_max, self.args.perspective_max)
                            four_point_org_augment[0, p, xy] += t1 # original for 256
                            four_point_1_augment[0, p, xy] += t1 * beta / alpha # original for 256 then to 512 in 1536 scale then to 256 in 1536 scale
                elif augment_type_single == "no":
                    pass
                else:
                    raise NotImplementedError()
            else:
                if self.args.rotate_max!=0:
                    rotation = torch.tensor(self.rng.random() - 0.5) * 2 * self.args.rotate_max # on 256x256
                    four_point_org_augment, four_point_1_augment = self.rotate_transform(rotation, four_point_org, four_point_1, four_point_org_augment, four_point_1_augment)
                elif self.args.resize_max!=0:
                    scale_factor = 1 + (self.rng.random() - 0.5) * 2 * self.args.resize_max # on 256x256
                    assert scale_factor > 0
                    four_point_org_augment, four_point_1_augment = self.resize_transform(scale_factor, beta, alpha, four_point_org_augment, four_point_1_augment)
                elif self.args.perspective_max!=0:
                    for p in range(4):
                        for xy in range(2):
                            t1 = self.rng.integers(-self.args.perspective_max, self.args.perspective_max) # on 256x256
                            four_point_org_augment[0, p, xy] += t1 # original for 256
                            four_point_1_augment[0, p, xy] += t1 * beta / alpha # original for 256 then to 512 in 1536 scale then to 256 in 1536 scale
                else:
                    raise NotImplementedError()
            H = tgm.get_perspective_transform(four_point_org, four_point_org_augment)
            H_inverse = torch.inverse(H)
            img1_width = img1.shape[1]
            four_point_raw = torch.zeros((2, 2, 2))
            top_left_raw = torch.Tensor([0, 0])
            top_right_raw = torch.Tensor([img1_width - 1, 0])
            bottom_left_raw = torch.Tensor([0, img1_width - 1])
            bottom_right_raw = torch.Tensor([img1_width - 1, img1_width - 1])
            four_point_raw[:, 0, 0] = top_left_raw
            four_point_raw[:, 0, 1] = top_right_raw
            four_point_raw[:, 1, 0] = bottom_left_raw
            four_point_raw[:, 1, 1] = bottom_right_raw
            four_point_raw = four_point_raw.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
            H_1 = tgm.get_perspective_transform(four_point_raw, four_point_org)
            H_1_inverse = torch.inverse(H_1)
            H_total = H_1_inverse @ H_inverse @ H_1
            img1 = tgm.warp_perspective(img1.unsqueeze(0), H_total, (img1_width, img1_width)).squeeze(0)
            four_point_1 = four_point_1_augment

        img1 = transforms.CenterCrop(self.args.crop_width)(img1)
        img1 = self.base_transform(img1)

        H = tgm.get_perspective_transform(four_point_org, four_point_1)
        H = H.squeeze()
        
        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H.numpy()).squeeze()
        diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((img1.shape[1], img1.shape[2]))
        diff_y_branch1 = diff_y_branch1.reshape((img1.shape[1], img1.shape[2]))
        pf_patch = np.zeros((self.args.resize_width, self.args.resize_width, 2))
        pf_patch[:, :, 0] = diff_x_branch1
        pf_patch[:, :, 1] = diff_y_branch1
        flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()
        H = H.squeeze()
        if neg_img2 is None:
            return img2, img1, flow, H, query_utm, database_utm, index, pos_index
        else:
            return img2, img1, flow, H, query_utm, database_utm, index, pos_index, neg_img2

class MYDATA(homo_dataset):
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super(MYDATA, self).__init__(args, augment= (args.augment == "img"))
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        exclude_val_region = args.exclude_val_region
        # Redirect datafolder path to h5
        # self.database_folder_h5_path = join(
        #     datasets_folder, dataset_name, split + "_database.h5"
        # )
        self.database_folder_nameh5_path = join(
            datasets_folder, dataset_name, split + "_database.h5"
        )
        self.database_folder_map_path = join(
            datasets_folder, "20201117_west_of_rimah", "20201117_west_of_rimah_BingSatellite.tif"
        )
        self.database_folder_map_df = F.to_tensor(Image.open(self.database_folder_map_path))
        self.queries_folder_h5_path = join(
            datasets_folder, dataset_name, split + "_queries.h5"
        )
        # database_folder_h5_df = h5py.File(self.database_folder_h5_path, "r", swmr=True)
        database_folder_nameh5_df = h5py.File(self.database_folder_nameh5_path, "r", swmr=True)
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r", swmr=True)

        # Map name to index
        self.database_name_dict = {}
        self.queries_name_dict = {}

        # Duplicated elements are added
        # for index, database_image_name in enumerate(database_folder_h5_df["image_name"]):
        for index, database_image_name in enumerate(database_folder_nameh5_df["image_name"]):
            database_image_name_decoded = database_image_name.decode("UTF-8")
            while database_image_name_decoded in self.database_name_dict:
                northing = [str(float(database_image_name_decoded.split("@")[2])+0.00001)]
                database_image_name_decoded = "@".join(list(database_image_name_decoded.split("@")[:2]) + northing + list(database_image_name_decoded.split("@")[3:]))
            self.database_name_dict[database_image_name_decoded] = index
        for index, queries_image_name in enumerate(queries_folder_h5_df["image_name"]):
            queries_image_name_decoded = queries_image_name.decode("UTF-8")
            while queries_image_name_decoded in self.queries_name_dict:
                northing = [str(float(queries_image_name_decoded.split("@")[2])+0.00001)]
                queries_image_name_decoded = "@".join(list(queries_image_name_decoded.split("@")[:2]) + northing + list(queries_image_name_decoded.split("@")[3:]))
            self.queries_name_dict[queries_image_name_decoded] = index

        # Read paths and UTM coordinates for all images.
        # database_folder = join(self.dataset_folder, "database")
        # queries_folder  = join(self.dataset_folder, "queries")
        # if not os.path.exists(database_folder): raise FileNotFoundError(f"Folder {database_folder} does not exist")
        # if not os.path.exists(queries_folder) : raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = sorted(self.database_name_dict)
        self.queries_paths = sorted(self.queries_name_dict)
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.database_paths]
        ).astype(np.float)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.queries_paths]
        ).astype(np.float)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,
            radius=args.val_positive_dist_threshold,
            return_distance=False,
        )

        # Find hard_negatives_per_query. Hard negative is out of prior position threshold and we don't care
        if args.prior_location_threshold != -1:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.hard_negatives_per_query = knn.radius_neighbors(
                self.queries_utms,
                radius=args.prior_location_threshold,
                return_distance=False,
            )

        # Add database, queries prefix
        for i in range(len(self.database_paths)):
            self.database_paths[i] = "database_" + self.database_paths[i]
        for i in range(len(self.queries_paths)):
            self.queries_paths[i] = "queries_" + self.queries_paths[i]

        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        # Close h5 and initialize for h5 reading in __getitem__
        # self.database_folder_h5_df = None
        self.database_folder_nameh5_df = None
        self.queries_folder_h5_df = None
        # database_folder_h5_df.close()
        database_folder_nameh5_df.close()
        queries_folder_h5_df.close()
        
        # Some queries might have no positive, we should remove those queries.
        queries_without_any_soft_positive = np.where(
            np.array([len(p)
                     for p in self.soft_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_soft_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_soft_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        # Remove queries without positives
        self.soft_positives_per_query = np.delete(self.soft_positives_per_query, queries_without_any_soft_positive)
        self.queries_paths = np.delete(self.queries_paths, queries_without_any_soft_positive)
        self.queries_utms = np.delete(self.queries_utms, queries_without_any_soft_positive, axis=0)

        # Remove queries in val region for extended dataset if necessary
        if exclude_val_region and self.split=="extended":
            queries_in_val_region = np.where(
                (self.queries_utms[:, 0] > TB_val_region[0])
                & (self.queries_utms[:, 0] < TB_val_region[2])
                & (self.queries_utms[:, 1] > TB_val_region[1])
                & (self.queries_utms[:, 1] < TB_val_region[3])
            )[0]
            if len(queries_in_val_region) != 0:
                logging.info(
                    f"There are {len(queries_in_val_region)} queries in the validation region "
                    + "within the extended set. They won't be considered because it will affect validation."
                )
            # Remove queries in val region
            self.soft_positives_per_query = np.delete(self.soft_positives_per_query, queries_in_val_region)
            self.queries_paths = np.delete(self.queries_paths, queries_in_val_region)
            self.queries_utms = np.delete(self.queries_utms, queries_in_val_region, axis=0)

        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)
        self.queries_num = len(self.queries_paths)
    
        if not os.path.isfile(f"cache/{self.split}_{args.val_positive_dist_threshold}_pairs.pth"):
            logging.info("Using online test pairs or generating test pairs. It is possible that different batch size can generate different test pairs.")
            print("Using online test pairs or generating test pairs. It is possible that different batch size can generate different test pairs.")
            self.test_pairs = None
        else:
            logging.info("Loading cached test pairs to make sure that the test pairs will not change for different batch size.")
            print("Loading cached test pairs to make sure that the test pairs will not change for different batch size.")
            self.test_pairs = torch.load(f"cache/{self.split}_{args.val_positive_dist_threshold}_pairs.pth")

    def get_positive_indexes(self, query_index):
        positive_indexes = self.soft_positives_per_query[query_index]
        return positive_indexes
     
    def __len__(self):
        return self.queries_num
    
    def __getitem__(self, index):
        # Init
        if self.queries_folder_h5_df is None:
            # self.database_folder_h5_df = h5py.File(
            #     self.database_folder_h5_path, "r", swmr=True)
            self.database_folder_nameh5_df = h5py.File(
                self.database_folder_nameh5_path, "r", swmr=True)
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r", swmr=True)
            
        # Queries
        if self.args.G_contrast!="none" and self.split!="extended":
            if self.args.G_contrast == "manual":
                img = transforms.functional.adjust_contrast(self._find_img_in_h5(index, database_queries_split="queries"), contrast_factor=3)
            elif self.args.G_contrast == "autocontrast":
                img = transforms.functional.autocontrast(self._find_img_in_h5(index, database_queries_split="queries"))
            elif self.args.G_contrast == "equalize":
                img =  transforms.functional.equalize(self._find_img_in_h5(index, database_queries_split="queries"))
            else:
                raise NotImplementedError()
        else:
            img = self._find_img_in_h5(index, database_queries_split="queries")
        
        # Positives
        if self.test_pairs is not None:
            pos_index = self.test_pairs[index]
        else:
            pos_index = random.choice(self.get_positive_indexes(index))
        # pos_img = self._find_img_in_h5(pos_index, database_queries_split="database")
        pos_img = self._find_img_in_map(pos_index, database_queries_split="database")
        
        query_utm = torch.tensor(self.queries_utms[index]).unsqueeze(0)
        database_utm = torch.tensor(self.database_utms[pos_index]).unsqueeze(0)
    
        return super(MYDATA, self).__getitem__(img, pos_img, query_utm, database_utm, index, pos_index)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def _find_img_in_h5(self, index, database_queries_split=None):
        # Find inside index for h5
        if database_queries_split is None:
            image_name = "_".join(self.images_paths[index].split("_")[1:])
            database_queries_split = self.images_paths[index].split("_")[0]
        else:
            if database_queries_split == "database":
                image_name = "_".join(
                    self.database_paths[index].split("_")[1:])
            elif database_queries_split == "queries":
                image_name = "_".join(self.queries_paths[index].split("_")[1:])
            else:
                raise KeyError("Dont find correct database_queries_split!")

        if database_queries_split == "database":
            img = Image.fromarray(
                self.database_folder_h5_df["image_data"][
                    self.database_name_dict[image_name]
                ]
            )
        elif database_queries_split == "queries":
            img = Image.fromarray(
                self.queries_folder_h5_df["image_data"][
                    self.queries_name_dict[image_name]
                ]
            )
        else:
            raise KeyError("Dont find correct database_queries_split!")

        return img
    
    def _find_img_in_map(self, index, database_queries_split=None):
        if database_queries_split != 'database':
            raise NotImplementedError()
        image_name = "_".join(
            self.database_paths[index].split("_")[1:])
        img = self.database_folder_map_df
        center_cood = (float(image_name.split("@")[1]), float(image_name.split("@")[2]))
        area = (int(center_cood[1]) - self.args.database_size//2, int(center_cood[0]) - self.args.database_size//2,
                int(center_cood[1]) + self.args.database_size//2, int(center_cood[0]) + self.args.database_size//2)
        img = F.crop(img=img, top=area[1], left=area[0], height=area[3]-area[1], width=area[2]-area[0])
        return img

class MYTRIPLETDATA(MYDATA):
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = "random" # default to random
        self.neg_dist_threshold = (args.database_size // 2 + args.crop_width // 2) * math.sqrt(2) # Outside of image

        # Find soft_negatives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_negatives_per_query = list(
            knn.radius_neighbors(
                self.queries_utms,
                radius=self.neg_dist_threshold,  # 10 meters
                return_distance=False,
            )
        )

    def recompute_negatives_random(self, args):
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        self.negative_samples = []
        for index in tqdm(range(self.queries_num)):
            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_negatives = self.soft_negatives_per_query[index]
            neg_indexes = np.random.choice(
                self.database_num,
                size=1,
                replace=False,
                )
            while neg_indexes in soft_negatives:
                neg_indexes = np.random.choice(
                self.database_num,
                size=1,
                replace=False,
                )
            self.negative_samples.append(
                neg_indexes
            )

        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.negative_samples = torch.tensor(self.negative_samples)

    def __getitem__(self, index):
        # Init
        if self.queries_folder_h5_df is None:
            # self.database_folder_h5_df = h5py.File(
            #     self.database_folder_h5_path, "r", swmr=True)
            self.database_folder_nameh5_df = h5py.File(
                self.database_folder_nameh5_path, "r", swmr=True)
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r", swmr=True)
            
        # Queries
        if self.args.G_contrast!="none" and self.split!="extended":
            if self.args.G_contrast == "manual":
                img = transforms.functional.adjust_contrast(self._find_img_in_h5(index, database_queries_split="queries"), contrast_factor=3)
            elif self.args.G_contrast == "autocontrast":
                img = transforms.functional.autocontrast(self._find_img_in_h5(index, database_queries_split="queries"))
            elif self.args.G_contrast == "equalize":
                img =  transforms.functional.equalize(self._find_img_in_h5(index, database_queries_split="queries"))
            else:
                raise NotImplementedError()
        else:
            img = self._find_img_in_h5(index, database_queries_split="queries")
        
        # Positives
        if self.test_pairs is not None and not self.args.generate_test_pairs:
            pos_index = self.test_pairs[index]
        else:
            pos_index = random.choice(self.get_positive_indexes(index))
        # pos_img = self._find_img_in_h5(pos_index, database_queries_split="database")
        pos_img = self._find_img_in_map(pos_index, database_queries_split="database")
        neg_img = self._find_img_in_map(self.negative_samples[index], database_queries_split="database")
        
        query_utm = torch.tensor(self.queries_utms[index]).unsqueeze(0)
        database_utm = torch.tensor(self.database_utms[pos_index]).unsqueeze(0)
    
        return homo_dataset.__getitem__(self, img, pos_img, query_utm, database_utm, index, pos_index, neg_img)

def fetch_dataloader(args, split='train'):
    train_dataset = MYDATA(args, args.datasets_folder, args.dataset_name, split)
    if split == 'train' or split == 'extended':
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                        pin_memory=True, shuffle=True, num_workers=args.num_workers,
                                        drop_last=True, worker_init_fn=seed_worker)
    elif split == 'val' or split == 'test':
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                pin_memory=True, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, worker_init_fn=seed_worker, generator=g)
    logging.info(f"{split} set: {train_dataset}")
    return train_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import kornia.geometry.transform as tgm

import random
from glob import glob
import os.path as osp
import cv2
from os.path import join
import h5py
from sklearn.neighbors import NearestNeighbors
import logging
from PIL import Image
import torchvision.transforms as transforms

marginal = 0
patch_size = 256

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

TB_val_region = [2650, 5650, 5100, 9500]

inv_base_transforms = transforms.Compose(
    [ 
        transforms.Normalize(mean = [ -m/s for m, s in zip(imagenet_mean, imagenet_std)],
                             std = [ 1/s for s in imagenet_std]),
        transforms.ToPILImage()
    ]
)

class homo_dataset(data.Dataset):
    def __init__(self, permute=False, resize_small=False):

        self.is_test = False
        self.init_seed = True
        self.image_list_img1 = []
        self.image_list_img2 = []
        self.dataset=[]
        self.permute = permute
        identity_transform = transforms.Lambda(lambda x: x)
        base_transform = transforms.Compose(
            [
                transforms.Resize([256, 256]) 
                if resize_small else identity_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        self.query_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                base_transform
            ]
        )
        self.database_transform = base_transform
        
    def __getitem__(self, query_PIL_image, database_PIL_image, query_utm, database_utm):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        img1 = query_PIL_image
        img2 = database_PIL_image # img1 warp to img2

        height, width = img1.size
        t = np.float32(np.array(query_utm - database_utm))
        t[0][0], t[0][1] = t[0][1], t[0][0] # Swap!
        
        if self.permute:
            raise NotImplementedError()
            y = random.randint(marginal, height - marginal - patch_size)
            x = random.randint(marginal, width - marginal - patch_size)
            
            top_left_point = (x, y)
            bottom_right_point = (patch_size + x, patch_size + y)

            perturbed_four_points_cord = []

            top_left_point_cord = (x, y)
            bottom_left_point_cord = (x, patch_size + y - 1)
            bottom_right_point_cord = (patch_size + x - 1, patch_size + y - 1)
            top_right_point_cord = (x + patch_size - 1, y)
            four_points_cord = [top_left_point_cord, bottom_left_point_cord, bottom_right_point_cord, top_right_point_cord]

            try:
                perturbed_four_points_cord = []
                for i in range(4):
                    t1 = random.randint(-marginal, marginal)
                    t2 = random.randint(-marginal, marginal)

                    perturbed_four_points_cord.append((four_points_cord[i][0] + t1,
                                                    four_points_cord[i][1] + t2))
                    
                y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
                point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()
                org = np.float32(four_points_cord) - t
                dst = np.float32(perturbed_four_points_cord) 
                H = cv2.getPerspectiveTransform(org, dst)
                H_inverse = np.linalg.inv(H)
            except:
                raise KeyError()
                # perturbed_four_points_cord = []
                # for i in range(4):
                #     t1 = 32//(i+1)
                #     t2 = -32//(i+1)

                #     perturbed_four_points_cord.append((four_points_cord[i][0] + t1,
                #                                       four_points_cord[i][1] + t2))

                # y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
                # point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

                # org = np.float32(four_points_cord)
                # dst = np.float32(perturbed_four_points_cord)
                # H = cv2.getPerspectiveTransform(org, dst)
                # H_inverse = np.linalg.inv(H)

            warped_image = cv2.warpPerspective(img2, H_inverse, (img1.shape[1], img1.shape[0]))
            # img1_pil = Image.fromarray(img1)
            # warped_pil = Image.fromarray(warped_image)
            # img1_pil.save(f"{database_utm}data.png")
            # warped_pil.save(f"{query_utm}query.png")
            img_patch_ori = img1[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0], :]
            img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0],:]

            point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H).squeeze()

            diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
            diff_x_branch1 = diff_branch1[:, 0]
            diff_y_branch1 = diff_branch1[:, 1]

            diff_x_branch1 = diff_x_branch1.reshape((img1.shape[0], img1.shape[1]))
            diff_y_branch1 = diff_y_branch1.reshape((img1.shape[0], img1.shape[1]))

            pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                                top_left_point[0]:bottom_right_point[0]]

            pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                                top_left_point[0]:bottom_right_point[0]]

            pf_patch = np.zeros((patch_size, patch_size, 2))
            pf_patch[:, :, 0] = pf_patch_x_branch1
            pf_patch[:, :, 1] = pf_patch_y_branch1

            img_patch_ori = img_patch_ori[:, :, ::-1].copy()
            img_patch_pert = img_patch_pert[:, :, ::-1].copy()
            img1 = torch.from_numpy((img_patch_ori)).float().permute(2, 0, 1)
            img2 = torch.from_numpy((img_patch_pert)).float().permute(2, 0, 1)
            flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()

            ### homo BUT NOT USED !!!!!!!!!!!!!
            four_point_org = torch.zeros((2, 2, 2))
            four_point_org[:, 0, 0] = torch.Tensor([0, 0])
            four_point_org[:, 0, 1] = torch.Tensor([patch_size - 1, 0])
            four_point_org[:, 1, 0] = torch.Tensor([0, patch_size - 1])
            four_point_org[:, 1, 1] = torch.Tensor([patch_size - 1, patch_size - 1])

            four_point = torch.zeros((2, 2, 2))
            four_point[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0])
            four_point[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([patch_size - 1, 0])
            four_point[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, patch_size - 1])
            four_point[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([patch_size - 1, patch_size - 1])
            four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
            four_point = four_point.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
            H = tgm.get_perspective_transform(four_point_org, four_point)
            H = H.squeeze()
        else:
            img1, img2 = self.query_transform(img1), self.database_transform(img2)
            if not self.args.resize_small:
                img1 = img1[:,128:384,128:384]
                img2 = img2[:,128:384,128:384]
            elif self.args.database_size == 512:
                t = t/2
            elif self.args.database_size == 1024:
                t = t/4
            elif self.args.database_size == 1536:
                t = t/6
            else:
                return NotImplementedError()
            
            t_tensor = torch.Tensor(t).squeeze(0)
            y_grid, x_grid = np.mgrid[0:img1.shape[1], 0:img1.shape[2]]
            point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()
            four_point_org = torch.zeros((2, 2, 2))
            top_left = torch.Tensor([0, 0])
            top_right = torch.Tensor([256 - 1, 0])
            bottom_left = torch.Tensor([0, 256 - 1])
            bottom_right = torch.Tensor([256 - 1, 256 - 1])
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
                top_left_resize = torch.Tensor([64, 64])
                top_right_resize = torch.Tensor([256 - 64 - 1, 64])
                bottom_left_resize = torch.Tensor([64, 256 - 64 - 1])
                bottom_right_resize = torch.Tensor([256 - 64 - 1, 256 - 64 - 1])
                four_point_1[:, 0, 0] = t_tensor + top_left_resize
                four_point_1[:, 0, 1] = t_tensor + top_right_resize
                four_point_1[:, 1, 0] = t_tensor + bottom_left_resize
                four_point_1[:, 1, 1] = t_tensor + bottom_right_resize
            elif self.args.database_size == 1536:
                top_left_resize2 = torch.Tensor([256/3, 256/3])
                top_right_resize2 = torch.Tensor([256 - 256/3 - 1, 256/3])
                bottom_left_resize2 = torch.Tensor([256/3, 256 - 256/3 - 1])
                bottom_right_resize2 = torch.Tensor([256 - 256/3 - 1, 256 - 256/3 - 1])
                four_point_1[:, 0, 0] = t_tensor + top_left_resize2
                four_point_1[:, 0, 1] = t_tensor + top_right_resize2
                four_point_1[:, 1, 0] = t_tensor + bottom_left_resize2
                four_point_1[:, 1, 1] = t_tensor + bottom_right_resize2
            four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
            four_point_1 = four_point_1.flatten(1).permute(1, 0).unsqueeze(0).contiguous() 
            H = tgm.get_perspective_transform(four_point_org, four_point_1)
            H = H.squeeze()
            
            point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H.numpy()).squeeze()
            diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
            diff_x_branch1 = diff_branch1[:, 0]
            diff_y_branch1 = diff_branch1[:, 1]

            diff_x_branch1 = diff_x_branch1.reshape((img1.shape[1], img1.shape[2]))
            diff_y_branch1 = diff_y_branch1.reshape((img1.shape[1], img1.shape[2]))
            pf_patch = np.zeros((256, 256, 2))
            pf_patch[:, :, 0] = diff_x_branch1
            pf_patch[:, :, 1] = diff_y_branch1
            flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()
            H = H.squeeze()
        return img2, img1, flow, H, query_utm, database_utm

class MYDATA(homo_dataset):
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", exclude_val_region=False):
        super(MYDATA, self).__init__(permute= (args.permute == "img"), resize_small=args.resize_small)
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        # Redirect datafolder path to h5
        self.database_folder_h5_path = join(
            datasets_folder, dataset_name, split + "_database.h5"
        )
        self.queries_folder_h5_path = join(
            datasets_folder, dataset_name, split + "_queries.h5"
        )
        database_folder_h5_df = h5py.File(self.database_folder_h5_path, "r", swmr=True)
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r", swmr=True)

        # Map name to index
        self.database_name_dict = {}
        self.queries_name_dict = {}

        # Duplicated elements are added
        for index, database_image_name in enumerate(database_folder_h5_df["image_name"]):
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
        self.database_folder_h5_df = None
        self.queries_folder_h5_df = None
        database_folder_h5_df.close()
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
    
    def get_positive_indexes(self, query_index):
        positive_indexes = self.soft_positives_per_query[query_index]
        return positive_indexes
     
    def __len__(self):
        return self.queries_num
    
    def __getitem__(self, index):
        # Init
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r", swmr=True)
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
        pos_index = random.choice(self.get_positive_indexes(index))
        pos_img = self._find_img_in_h5(pos_index, database_queries_split="database")
        
        query_utm = torch.tensor(self.queries_utms[index]).unsqueeze(0)
        database_utm = torch.tensor(self.database_utms[pos_index]).unsqueeze(0)
    
        return super(MYDATA, self).__getitem__(img, pos_img, query_utm, database_utm)

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

def fetch_dataloader(args, split='train', exclude_val_region=False):
    train_dataset = MYDATA(args, args.datasets_folder, args.dataset_name, split, exclude_val_region=exclude_val_region)
    if split == 'train' or split == 'extended':
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                        pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    elif split == 'val' or split == 'test':
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                pin_memory=True, shuffle=False, num_workers=8,
                                drop_last=False, worker_init_fn=seed_worker, generator=g)
    logging.info(f"{split} set: {train_dataset}")
    return train_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
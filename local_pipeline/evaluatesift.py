import sys

sys.path.append('core')
import parser
from PIL import Image
import argparse
import os
import cv2
import time
from os.path import join
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import datasets_4cor_img as datasets
from datasets_4cor_img import inv_base_transforms
from utils import save_overlap_img
import logging
from utils import count_parameters, save_img, save_overlap_img, setup_seed, warp
import commons
import wandb
from uuid import uuid4



def mysift(img1, img2):
    MIN_MATCH_COUNT = 10
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # img_small_sift = cv2.drawKeypoints(img1, kp1, outImage=np.array([]),
    #                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img_big_sift = cv2.drawKeypoints(img2, kp2, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("img1", img_small_sift)
    # cv2.imshow("img2", img_big_sift)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # filter out matches with low score
    good = []
    for m, n in matches:
        if m.distance <= 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w, dim = img1.shape
        # pts is the four vertices of img1
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # four vertices after transformation
        dst = cv2.perspectiveTransform(pts, M)

        # img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d", (len(good), MIN_MATCH_COUNT))
        matchesMask = None



@torch.no_grad()
def validate_process(args):
    """ Perform evaluation on the FlyingChairs (test) split """

    total_mace = torch.empty(0)
    total_flow = torch.empty(0)
    total_ce = torch.empty(0)
    total_mace_conf_error = torch.empty(0)
    timeall = []
    mace_conf_list = []
    mace_conf_error_list = []
    val_loader = datasets.fetch_dataloader(args, split='val')
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        img1, img2, flow_gt, H, query_utm, database_utm, image1_ori = [x for x in data_blob]

        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")

        if i_batch%1000 == 0:
            save_img(torchvision.utils.make_grid((img1)),
                     args.save_dir + "/b1_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.png')
            save_img(torchvision.utils.make_grid((img2)),
                     args.save_dir + "/b2_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.png')

        if not args.identity:
            # model.set_input(img1, img2, flow_gt, image1_ori)
            flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            flow_ = (flow_4cor)**2
            flow_ = ((flow_[:,0,:,:] + flow_[:,1,:,:])**0.5)
            flow_vec = torch.mean(torch.mean(flow_, dim=1), dim=1)


        img1 = cv2.cvtColor(np.array(inv_base_transforms(img1.squeeze().detach().cpu())), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(inv_base_transforms(img2.squeeze().detach().cpu())), cv2.COLOR_RGB2BGR)
        image1_ori = cv2.cvtColor(np.array(inv_base_transforms(image1_ori.squeeze().detach().cpu())), cv2.COLOR_RGB2BGR)


        time_start = time.time()

        if args.train_ue_method != 'train_only_ue_raw_input':
            if not args.identity:
                mysift(img1, img2)

                # model.forward(use_raw_input=(args.train_ue_method == 'train_only_ue_raw_input'), noise_std=args.noise_std, sample_method=args.sample_method)
                # four_pred = model.four_pred
                time_end = time.time()
                timeall.append(time_end-time_start)
                # print(time_end-time_start)
            else:
                four_pred = torch.zeros((flow_gt.shape[0], 2, 2, 2))

        # img = inv_base_transforms(image1_ori.squeeze().detach().cpu())
        # plt.imshow(img)
        # plt.show()
        # print(image1.min(), image1.max(), image2.min(), image2.max())
        


if __name__ == "__main__":
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join(
        "logs",
        args.save_dir,
        f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
    )
    commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)

    # wandb.init(project="STGL-IHN", entity="xjh19971", config=vars(args))

    validate_process(args)
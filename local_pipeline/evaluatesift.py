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
import kornia.geometry.transform as tgm
import logging
from utils import count_parameters, save_img, save_overlap_img, setup_seed, warp
import commons
import wandb
from uuid import uuid4



def mysift(img1, img2, conf):
    MIN_MATCH_COUNT = 10

    if conf.points_detector == 'sift':
        # sift = cv2.xfeatures2d.SIFT_create() this is an old free version
        pd = cv2.SIFT_create()
    elif conf.points_detector == 'orb':
        pd = cv2.ORB_create(nfeatures=500)
    elif conf.points_detector == 'brisk':
        pd = cv2.BRISK_create()

    kp1, des1 = pd.detectAndCompute(img1, None)
    kp2, des2 = pd.detectAndCompute(img2, None)

    # matching
    try: # many pair of images have no matching. If no matching, then return False.
        if conf.points_detector == 'sift':
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)

        elif conf.points_detector == 'orb':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2

            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)
            # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            # matches = matcher.match(des1, des2)
            # print(matches)

        elif conf.points_detector == 'brisk':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

            matches = matcher.knnMatch(des1, des2, k=2)
            # print(matches)

    except Exception as e:
        return False, []

    # filter out matches with low score
    good = []
    dst = []

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance <= conf.match_threshold * n.distance:
                good.append(m)
        except ValueError:
            pass

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float64([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float64([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if conf.solver == 'ransac':
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        elif conf.solver == 'magsac++':
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)

        if M is not None:
            matchesMask = mask.ravel().tolist()

            if len(img1.shape) == 3:
                h, w, dim = img1.shape
            elif len(img1.shape) == 2:
                h, w = img1.shape
            # pts is the four vertices of img1
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # four vertices after transformation
            dst = cv2.perspectiveTransform(pts, M)  # shape (4, 1, 2)
            dst = pts - dst
            # print(dst)

            four_pred = np.zeros((conf.batch_size, 2, 2, 2))
            four_pred[:, :, 0, 0] = dst[0, 0, :]
            four_pred[:, :, 0, 1] = dst[3, 0, :]
            four_pred[:, :, 1, 0] = dst[1, 0, :]
            four_pred[:, :, 1, 1] = dst[2, 0, :]


            # uncomment to visualize extracted keypoints
            # img1_kp = cv2.drawKeypoints(img1, kp1, outImage=np.array([]),
            #                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # img2_kp = cv2.drawKeypoints(img2, kp2, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("img1", img1_kp)
            # cv2.imshow("img2", img2_kp)
            # cv2.waitKey()
            # cv2.destroyAllWindows()


            # uncomment to visualize the homography transformation
            # img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
            #                    singlePointColor=None,
            #                    matchesMask=matchesMask,  # draw only inliers
            #                    flags=2)
            # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
            
            # plt.figure(figsize=(5, 5))
            # plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            # plt.show()

            return True, four_pred
        else:
            return False, dst

    else:
        return False, dst



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
    fail_count = 0
    success_count = 0
    val_loader = datasets.fetch_dataloader(args, split='test')
    for i_batch, data_blob in enumerate(val_loader):
        # for i_batch, data_blob in enumerate(tqdm(val_loader)):
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


        img1 = cv2.cvtColor(np.array(inv_base_transforms(img1.squeeze().detach().cpu())), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(inv_base_transforms(img2.squeeze().detach().cpu())), cv2.COLOR_RGB2GRAY)
        image1_ori = cv2.cvtColor(np.array(inv_base_transforms(image1_ori.squeeze().detach().cpu())), cv2.COLOR_RGB2GRAY)

        # img1 = cv2.imread('../datasets/1.png', cv2.COLOR_BGR2GRAY)
        # img2 = cv2.imread('../datasets/2.png', cv2.COLOR_BGR2GRAY)

        time_start = time.time()

        if args.train_ue_method != 'train_only_ue_raw_input':
            if not args.identity:
                flag, four_pred = mysift(img1, img2, conf=args)
                time_end = time.time()
                timeall.append(time_end - time_start)
                # print(time_end-time_start)

                if not flag:
                    fail_count += 1  # the matching is not successful, continue to the next image.
                    # pbar.set_postfix_str(f"unmatched: {fail_count}, matched: {success_count}")
                    # pbar.update(1)
                    continue
                else:
                    success_count += 1
                    # pbar.set_postfix_str(f"unmatched: {fail_count}, matched: {success_count}")
                    # pbar.update(1)

                mace_ = (flow_4cor - torch.from_numpy(four_pred)) ** 2
                mace_ = ((mace_[:, 0, :, :] + mace_[:, 1, :, :]) ** 0.5)
                mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
                # print(mace_vec)
                total_mace = torch.cat([total_mace, mace_vec], dim=0)
                final_mace = torch.mean(total_mace).item()
                total_flow = torch.cat([total_flow, flow_vec], dim=0)
                final_flow = torch.mean(total_flow).item()

                # CE
                four_point_org_single = torch.zeros((1, 2, 2, 2))
                four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0])
                four_point_org_single[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
                four_point_org_single[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
                four_point_org_single[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])
                four_point_1 = torch.from_numpy(four_pred) + four_point_org_single
                four_point_org = four_point_org_single.repeat(four_point_1.shape[0], 1, 1, 1).flatten(2).permute(0, 2,
                                                                                                                    1).contiguous()
                four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
                four_point_gt = flow_4cor + four_point_org_single
                four_point_gt = four_point_gt.flatten(2).permute(0, 2, 1).contiguous()

                H = tgm.get_perspective_transform(four_point_org, four_point_1.float())
                center_T = torch.tensor([args.resize_width / 2 - 0.5, args.resize_width / 2 - 0.5, 1]).unsqueeze(
                    1).unsqueeze(0).repeat(H.shape[0], 1, 1)
                w = torch.bmm(H, center_T).squeeze(2)
                center_pred_offset = w[:, :2] / w[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)
                alpha = args.database_size / args.resize_width
                center_gt_offset = (query_utm - database_utm).squeeze(1) / alpha
                temp = center_gt_offset[:, 0].clone()
                center_gt_offset[:, 0] = center_gt_offset[:, 1]
                center_gt_offset[:, 1] = temp  # Swap!
                ce_ = (center_pred_offset - center_gt_offset) ** 2
                ce_ = ((ce_[:, 0] + ce_[:, 1]) ** 0.5)
                ce_vec = ce_
                total_ce = torch.cat([total_ce, ce_vec], dim=0)
                final_ce = torch.mean(total_ce).item()

            else:
                four_pred = torch.zeros((flow_gt.shape[0], 2, 2, 2))


            # if not args.identity:
            #     if args.use_ue:
            #         with torch.no_grad():
            #             conf_pred = model.predict_uncertainty(GAN_mode=args.GAN_mode)
            #         conf_vec = torch.mean(conf_pred, dim=[1, 2, 3])
            #         if args.GAN_mode == "macegan" and args.D_net != "ue_branch":
            #             logging.debug(f"conf_pred_diff:{conf_vec.cpu() - torch.exp(args.ue_alpha * mace_vec)}.")
            #             logging.debug(f"pred_mace:{mace_vec}")
            #             mace_conf_error_vec = F.l1_loss(conf_vec.cpu(), torch.exp(args.ue_alpha * mace_vec))
            #         elif args.GAN_mode == "vanilla_rej":
            #             flow_bool = torch.ones_like(flow_vec)
            #             alpha = args.database_size / args.resize_width
            #             flow_bool[flow_vec >= (args.rej_threshold / alpha)] = 0.0
            #             mace_conf_error_vec = F.binary_cross_entropy(conf_vec.cpu(), flow_bool) # sigmoid in predict uncertainty
            #         total_mace_conf_error = torch.cat([total_mace_conf_error, mace_conf_error_vec.reshape(1)], dim=0)
            #         final_mace_conf_error = torch.mean(total_mace_conf_error).item()
            #         if args.GAN_mode == "macegan" and args.D_net != "ue_branch":
            #             for i in range(len(mace_vec)):
            #                 mace_conf_list.append((mace_vec[i].item(), conf_vec[i].item()))
            #         elif args.GAN_mode == "vanilla_rej":
            #             for i in range(len(flow_vec)):
            #                 mace_conf_list.append((flow_vec[i].item(), conf_vec[i].item()))

    if not args.train_ue_method == "train_only_ue_raw_input":
        logging.info(f"MACE Metric: {final_mace}")
        logging.info(f'CE Metric: {final_ce}')
        logging.info(f'Number of failed matches: {fail_count} (of {val_loader.__len__()})')
        print(f"MACE Metric: {final_mace}")
        print(f'CE Metric: {final_ce}')
        print(f'Number of failed matches: {fail_count} (of {val_loader.__len__()})')


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
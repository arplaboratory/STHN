import numpy as np
import os
import torch
import argparse
from model.network import KeyNet
from utils import save_overlap_img, save_img, setup_seed, save_overlap_bbox_img
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
from tqdm import tqdm
import cv2
import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
from plot_hist import plot_hist_helper
import torch.nn.functional as F
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import wandb
import kornia.feature as KF
import kornia as K
from datasets_4cor_img import inv_base_transforms

def load_model(args, model):
    model_med = torch.load(args.eval_model, map_location='cuda:0')
    model.netG.load_state_dict(model_med['netG'], strict=True)
    model.setup() 
    model.netG.eval()
    return model

def test(args, wandb_log):
    model = KeyNet(args)
    # model = load_model(args, model)
    if args.test:
        val_dataset = datasets.fetch_dataloader(args, split='test')
    else:
        val_dataset = datasets.fetch_dataloader(args, split='val')
    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)
    
def evaluate_SNet(model, val_dataset, batch_size=0, args = None, wandb_log=False):

    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)
    total_flow = torch.empty(0)
    total_ce = torch.empty(0)
    args.ue_rej_std = [1]
    total_ue_mask = torch.empty(0, len(args.ue_rej_std))
    timeall=[]
        
    matcher = KF.LoFTR(pretrained="outdoor").cuda()
    
    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        img1, img2, flow_gt,  H, query_utm, database_utm, index, pos_index  = [x for x in data_blob]

        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")

        model.set_input(img1, img2, flow_gt)
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        four_point_org_single = torch.zeros((1, 2, 2, 2))
        four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0])
        four_point_org_single[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
        four_point_org_single[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
        four_point_org_single[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])
        four_point_augment = four_point_org_single[0].view(2, -1)
        four_point_augment = torch.cat([four_point_augment, torch.ones((1, four_point_augment.shape[1])).to(four_point_augment.device)], dim=0)
        image_1_raw = np.array(inv_base_transforms(model.image_1[0].clone().cpu()))
        image_2_raw = np.array(inv_base_transforms(model.image_2[0].clone().cpu()))
        with torch.inference_mode():
            if not hasattr(model, "imagenet_mean") or model.imagenet_mean is None:
                model.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(model.image_1.device)
                model.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(model.image_1.device)
            model.image_1 = (model.image_1.contiguous() - model.imagenet_mean) / model.imagenet_std
            model.image_2 = (model.image_2.contiguous() - model.imagenet_mean) / model.imagenet_std
            input_dict ={"image0": K.color.rgb_to_grayscale(model.image_2), "image1": K.color.rgb_to_grayscale(model.image_1)}
            correspondences = matcher(input_dict)
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()
            if len(mkpts0) > 10:
                try:
                    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
                    four_point_pred_augment = H @ four_point_augment.cpu().numpy()
                except Exception as e:
                    print("Skip: Cannot find H")
                    ue_mask = torch.zeros((1, 1), dtype=torch.bool)
                    total_ue_mask = torch.cat([total_ue_mask, ue_mask], dim=0)
                    continue
                four_point_pred = four_point_pred_augment[:2, :] / four_point_pred_augment[2, :]
                four_pred = torch.tensor(four_point_pred) - four_point_org_single[0].view(2, -1)
                four_pred = (four_pred.view(1, 2, 2, 2)).float()
                ue_mask = torch.ones((1, 1), dtype=torch.bool)
            else:
                ue_mask = torch.zeros((1, 1), dtype=torch.bool)
        try:
            mace_ = (flow_4cor - four_pred.cpu().detach())**2
            mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
            mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
            # print(mace_[0,:])
            ue_mask = torch.ones((mace_vec.shape[0], len(args.ue_rej_std)))
            model_eval = model
            
            # CE
            four_point_1 = four_pred.cpu().detach() + four_point_org_single
            four_point_org = four_point_org_single.repeat(four_point_1.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
            four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
            four_point_gt = flow_4cor.cpu().detach() + four_point_org_single
            four_point_gt = four_point_gt.flatten(2).permute(0, 2, 1).contiguous()
            H = tgm.get_perspective_transform(four_point_org, four_point_1)
            center_T = torch.tensor([args.resize_width/2-0.5, args.resize_width/2-0.5, 1]).unsqueeze(1).unsqueeze(0).repeat(H.shape[0], 1, 1)
            w = torch.bmm(H, center_T).squeeze(2)
            center_pred_offset = w[:, :2]/w[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)
            # alpha = args.database_size / args.resize_width
            # center_gt_offset = (query_utm - database_utm).squeeze(1) / alpha
            # temp = center_gt_offset[:, 0].clone()
            # center_gt_offset[:, 0] = center_gt_offset[:, 1]
            # center_gt_offset[:, 1] = temp # Swap!
            H_gt = tgm.get_perspective_transform(four_point_org, four_point_gt)
            w_gt = torch.bmm(H_gt, center_T).squeeze(2)
            center_gt_offset = w_gt[:, :2]/w_gt[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)
            ce_ = (center_pred_offset - center_gt_offset)**2
            ce_ = ((ce_[:,0] + ce_[:,1])**0.5)
            ce_vec = ce_
        except Exception as e:
            print(f"Skip: {e}")
            continue

        total_ue_mask = torch.cat([total_ue_mask, ue_mask], dim=0)
        total_mace = torch.cat([total_mace,mace_vec], dim=0)
        total_ce = torch.cat([total_ce, ce_vec], dim=0)
        
        print(mace_vec)
        print(ce_vec)
        # if args.vis_all:
        #     save_dir = os.path.join(args.save_dir, 'vis')
        #     if not os.path.exists(save_dir):
        #         os.mkdir(save_dir)
        #     if not args.two_stages:
        #         save_overlap_bbox_img(model_eval.image_1, model_eval.fake_warped_image_2, save_dir + f'/train_overlap_bbox_{i_batch}.png', four_point_gt, four_point_1, ue_mask=ue_mask)
        #     else:
        #         four_point_org_single_ori = torch.zeros((1, 2, 2, 2))
        #         four_point_org_single_ori[:, :, 0, 0] = torch.Tensor([0, 0])
        #         four_point_org_single_ori[:, :, 0, 1] = torch.Tensor([args.database_size - 1, 0])
        #         four_point_org_single_ori[:, :, 1, 0] = torch.Tensor([0, args.database_size - 1])
        #         four_point_org_single_ori[:, :, 1, 1] = torch.Tensor([args.database_size - 1, args.database_size - 1])
        #         four_point_bbox = model_eval.flow_bbox.cpu().detach() + four_point_org_single_ori
        #         alpha = args.database_size / args.resize_width
        #         four_point_bbox = four_point_bbox.flatten(2).permute(0, 2, 1).contiguous() / alpha
        #         save_overlap_bbox_img(model_eval.image_1, model_eval.fake_warped_image_2, save_dir + f'/train_overlap_bbox_{i_batch}.png', four_point_gt, four_point_1, crop_bbox=four_point_bbox, ue_mask=ue_mask)
        #         if args.ue_method == "augment":
        #             four_point_gt_multi = four_point_gt.repeat(args.ue_num_crops, 1, 1)
        #             four_point_1_multi = four_point_1.repeat(args.ue_num_crops, 1, 1)
        #             save_overlap_bbox_img(model_eval.image_1_multi, model_eval.fake_warped_image_2_multi_before, save_dir + f'/train_overlap_bbox_before_recover_{i_batch}.png', four_point_gt_multi, four_point_1_multi)
        #             save_overlap_bbox_img(model_eval.image_1_multi, model_eval.fake_warped_image_2_multi_after, save_dir + f'/train_overlap_bbox_after_recover_{i_batch}.png', four_point_gt_multi, four_point_1_multi)
    for j in range(total_ue_mask.shape[1]):
        ue_mask_single = total_ue_mask[:,j]
        final_ue_mask = torch.count_nonzero(ue_mask_single)/len(ue_mask_single)
        print(ue_mask_single.shape)
        print(total_mace.shape)
        print(total_ce.shape)
        final_mace = torch.mean(total_mace).item()
        final_ce = torch.mean(total_ce).item()
        logging.info(f"MACE Metric {j}: {final_mace}")
        logging.info(f'CE Metric {j}: {final_ce}')
        logging.info(f'Success rate {j}:{final_ue_mask}')
        print(f"MACE Metric {j}: {final_mace}")
        print(f'CE Metric {j}: {final_ce}')
        print(f'Success rate {j}:{final_ue_mask}')
        if wandb_log:
            wandb.log({f"test_mace_{j}": final_mace})
            wandb.log({f"test_ce_{j}": final_ce})
            wandb.log({f"success_rate_{j}": final_ue_mask})
    logging.info(np.mean(np.array(timeall[1:-1])))
    io.savemat(args.save_dir + '/resmat', {'matrix': total_mace.numpy()})
    np.save(args.save_dir + '/resnpy.npy', total_mace.numpy())
    plot_hist_helper(args.save_dir)

if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join(
    "test",
    args.save_dir,
    args.eval_model.split("/")[-2],
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    logging.debug(args)
    wandb_log = True
    if wandb_log:
        wandb.init(project="UAGL-eval", entity="xjh19971", config=vars(args))
    test(args, wandb_log)
import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import datasets_4cor_img as datasets
from utils import save_overlap_img
import logging

@torch.no_grad()
def validate_process(model, args, total_steps):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.netG.eval()
    if args.use_ue and args.D_net != "ue_branch":
        model.netD.eval()
    if args.two_stages:
        model.netG_fine.eval()
    mace_list = []
    mace_conf_list = []
    mace_conf_error_list = []
    val_loader = datasets.fetch_dataloader(args, split='val')
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
            
        image1, image2, flow_gt,  H, query_utm, database_utm, image1_ori  = [x for x in data_blob]
        
        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")
            
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        image1 = image1.to(model.netG.module.device)
        image2 = image2.to(model.netG.module.device)
        model.set_input(image1, image2, flow_gt, image1_ori)
        model.forward(use_raw_input=(args.train_ue_method == 'train_only_ue_raw_input'), noise_std=args.noise_std, sample_method=args.sample_method)
        if args.train_ue_method != 'train_only_ue_raw_input':
            if i_batch == 0:
                # Visualize
                save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                                torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                                args.save_dir + '/val_overlap_pred.png')
                save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                                torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                                args.save_dir + '/val_overlap_gt.png')
                if args.two_stages:
                    save_overlap_img(torchvision.utils.make_grid(model.image_1_crop, nrow=16, padding = 16, pad_value=0),
                                torchvision.utils.make_grid(model.image_2, nrow=16, padding = 16, pad_value=0), 
                                args.save_dir + '/val_overlap_crop.png')
            four_pr = model.four_pred
            mace = torch.sum((four_pr.cpu().detach() - flow_4cor) ** 2, dim=1).sqrt()
            mace_list.append(mace.view(-1).numpy())
        else:
            mace_list.append(np.zeros((image1.shape[0])))
        if args.use_ue:
            if args.GAN_mode =="macegan":
                mace_pred_vec =  torch.mean(torch.mean(mace, dim=1), dim=1)
            elif args.GAN_mode == "vanilla_rej":
                flow = (flow_4cor)**2
                flow = ((flow[:,0,:,:] + flow[:,1,:,:])**0.5)
                flow_vec = torch.mean(torch.mean(flow, dim=1), dim=1)
            conf_pred = model.predict_uncertainty(GAN_mode=args.GAN_mode, for_training=True)
            if args.D_net == "ue_branch":
                conf_pred_vec = torch.mean(conf_pred, dim=[1, 2])
            else:
                conf_pred_vec = torch.mean(conf_pred, dim=[1, 2, 3])
            if args.GAN_mode == "macegan":
                mace_conf_error = F.l1_loss(conf_pred_vec.cpu(), torch.exp(args.ue_alpha * torch.mean(torch.mean(mace, dim=1), dim=1)))
            elif args.GAN_mode == "vanilla_rej":
                flow_bool = torch.ones_like(flow_vec)
                alpha = args.database_size / args.resize_width
                flow_bool[flow_vec >= (args.rej_threshold / alpha)] = 0.0
                mace_conf_error = F.binary_cross_entropy_with_logits(conf_pred_vec.cpu(), flow_bool, pos_weight=torch.tensor(args.bce_weight)) # sigmoid in predict uncertainty
            mace_conf_error_list.append(mace_conf_error.numpy())
            if args.GAN_mode == "macegan":
                for i in range(len(mace_pred_vec)):
                    mace_conf_list.append((mace_pred_vec[i].item(), conf_pred_vec[i].item()))
            elif args.GAN_mode == "vanilla_rej":
                for i in range(len(flow_vec)):
                    mace_conf_list.append((flow_vec[i].item(), conf_pred_vec[i].item()))

    if args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
        model.netG.eval()
        if args.two_stages:
            model.netG_fine.eval()
    else:
        if args.restore_ckpt is None or args.finetune:
            model.netG.train()
        else:
            model.netG.eval()
        if args.two_stages:
            model.netG_fine.train()
    if args.use_ue and args.D_net != "ue_branch":
        model.netD.train()
        mace_conf_list = np.array(mace_conf_list)
        # plot mace conf
        plt.figure()
        # plt.axis('equal')
        plt.scatter(mace_conf_list[:,0], mace_conf_list[:,1], s=5)
        plt.xlabel("MACE")
        plt.ylabel("conf")
        plt.savefig(args.save_dir + f'/{total_steps}_conf.png')
        plt.close()
    mace = np.mean(np.concatenate(mace_list))
    mace_conf_error = np.mean(np.array(mace_conf_error_list)) if args.use_ue else 0
    logging.info("Validation MACE: %f" % mace)
    logging.info("Validation MACE CONF ERROR: %f" % mace_conf_error)
    return {'val_mace': mace, 'val_mace_conf_error': mace_conf_error}
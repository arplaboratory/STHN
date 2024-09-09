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

def validate_process(model, args, total_steps):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.netG.eval()
    if args.two_stages:
        model.netG_fine.eval()
    mace_list = []
    val_loader = datasets.fetch_dataloader(args, split='val')
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
            
        image1, image2, flow_gt,  H, query_utm, database_utm, _, _  = [x for x in data_blob]
        
        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")
            
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        if hasattr(model.netG, "module"):
            device = model.netG.module.device
        else:
            device = model.netG.device
        image1 = image1.to(device)
        image2 = image2.to(device)
        model.set_input(image1, image2, flow_gt)
        with torch.no_grad():
            model.forward()
        # if i_batch == 0:
        #     # Visualize
        #     save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
        #                     torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0), 
        #                     args.save_dir + '/val_overlap_pred.png')
        #     save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
        #                     torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
        #                     args.save_dir + '/val_overlap_gt.png')
        #     if args.two_stages:
        #         save_overlap_img(torchvision.utils.make_grid(model.image_1_crop, nrow=16, padding = 16, pad_value=0),
        #                     torchvision.utils.make_grid(model.image_2, nrow=16, padding = 16, pad_value=0), 
        #                     args.save_dir + '/val_overlap_crop.png')
        four_pr = model.four_pred
        mace = torch.sum((four_pr.cpu().detach() - flow_4cor) ** 2, dim=1).sqrt()
        mace_list.append(mace.view(-1).numpy())
    model.netG.train()
    if args.two_stages:
        model.netG_fine.train()
    mace = np.mean(np.concatenate(mace_list))
    logging.info("Validation MACE: %f" % mace)
    return {'val_mace': mace}
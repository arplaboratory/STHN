from __future__ import print_function, division
import json
import sys
import argparse
import os
import cv2
import time
import torch
import torchvision
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from model.network import STHEGAN, mywarp
from utils import count_parameters, save_img, save_overlap_img, setup_seed, warp
import commons
from os.path import join
from evaluate import validate_process
import datasets_4cor_img as datasets
import parser
import wandb
from datetime import datetime
from uuid import uuid4
import logging
from myevaluate import evaluate_SNet

def main(args):
    model = STHEGAN(args, for_training=True)
    model.setup()
    if args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
        model.netG.eval()
        for param in model.netG.parameters():
            param.requires_grad = False
    else:
        model.netG.train()
    logging.info(f"Parameter Count: {count_parameters(model.netG)}")
    if args.use_ue:
        model.netD.train()
        logging.info(f"Parameter Count: {count_parameters(model.netD)}")

    if args.restore_ckpt is not None:

        save_model = torch.load(args.restore_ckpt)
        
        model.netG.load_state_dict(save_model['netG'])
        if save_model['netD'] is not None:
            model.netD.load_state_dict(save_model['netD'])
        
    train_loader = datasets.fetch_dataloader(args, split="train")
    if os.path.exists(os.path.join(args.datasets_folder, args.dataset_name, "extended_queries.h5")):
        extended_loader = datasets.fetch_dataloader(args, split="extended", exclude_val_region=args.exclude_val_region)
    else:
        extended_loader = None

    total_steps = 0
    while total_steps <= args.num_steps:
        total_steps = train(model, train_loader, args, total_steps)
        if extended_loader is not None:
            total_steps = train(model, extended_loader, args, total_steps, train_step_limit=len(train_loader))

    test_dataset = datasets.fetch_dataloader(args, split='test')
    model_med = torch.load(args.save_dir + f'/{args.name}.pth')
    model.netG.load_state_dict(model_med['netG'])
    if args.use_ue:
        model.netD.load_state_dict(model_med['netD'])
    evaluate_SNet(model, test_dataset, batch_size=args.batch_size, args=args, wandb_log=True)

def train(model, train_loader, args, total_steps, train_step_limit = None):
    count = 0
    last_best_val_mace = None
    last_best_val_mace_conf_error = None
    top_left = torch.Tensor([0, 0]).to(model.netG.module.device)
    top_right = torch.Tensor([256 - 1, 0]).to(model.netG.module.device)
    bottom_left = torch.Tensor([0, 256 - 1]).to(model.netG.module.device)
    bottom_right = torch.Tensor([256 - 1, 256 - 1]).to(model.netG.module.device)
    four_point_org_single = torch.zeros((1, 2, 2, 2)).to(model.netG.module.device)
    four_point_org_single[:, :, 0, 0] = top_left
    four_point_org_single[:, :, 0, 1] = top_right
    four_point_org_single[:, :, 1, 0] = bottom_left
    four_point_org_single[:, :, 1, 1] = bottom_right
    
    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        tic = time.time()
        image1, image2, flow, _, query_utm, database_utm, image1_ori  = [x.cuda() for x in data_blob]
        # image2_w = warp(image2, flow)
        model.set_input(image1, image2, flow, image1_ori)
        if i_batch==0:
            save_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img1.bmp')
            save_img(torchvision.utils.make_grid(image2, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img2.bmp')
            save_img(torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img2w.bmp')
            save_overlap_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0),
                             torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                             args.save_dir + '/train_overlap.png')
        metrics = model.optimize_parameters()
        model.update_learning_rate()
        metrics["lr"] = model.scheduler_G.get_lr()
        toc = time.time()
        metrics['time'] = toc - tic
        wandb.log({
                "mace": metrics["mace"] if args.train_ue_method == 'train_end_to_end' else 0,
                "lr": metrics["lr"],
                "G_loss": metrics["G_loss"] if args.train_ue_method == 'train_end_to_end' else 0,
                "GAN_loss": metrics["GAN_loss"] if args.train_ue_method == 'train_end_to_end' and args.use_ue else 0,
                "D_loss": metrics["D_loss"] if args.use_ue else 0
            },)
        total_steps += 1
        # Validate
        if total_steps % args.val_freq == args.val_freq - 1:
            current_val_mace, current_val_mace_conf_error = validate(model, args, total_steps)
            # plot_train(logger, args)
            # plot_val(logger, args)
            PATH = args.save_dir + f'/{total_steps+1}_{args.name}.pth'
            checkpoint = {
                "netG": model.netG.state_dict(),
                "netD": model.netD.state_dict() if args.use_ue else None,
            }
            torch.save(checkpoint, PATH)
            if args.use_ue and args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
                if last_best_val_mace_conf_error is None or last_best_val_mace_conf_error > current_val_mace_conf_error:
                    last_best_val_mace_conf_error = current_val_mace_conf_error
                    PATH = args.save_dir + f'/{args.name}.pth'
                    torch.save(checkpoint, PATH)
            else:
                if last_best_val_mace is None or last_best_val_mace > current_val_mace:
                    logging.info(f"Saving best model, last_best_val_mace: {last_best_val_mace}, current_val_mace: {current_val_mace}")
                    last_best_val_mace = current_val_mace
                    PATH = args.save_dir + f'/{args.name}.pth'
                    torch.save(checkpoint, PATH)
                else:
                    logging.info(f"No Saving, last_best_val_mace: {last_best_val_mace}, current_val_mace: {current_val_mace}")

        if total_steps >= args.num_steps:
            break
        
        if train_step_limit is not None and count >= train_step_limit: # Balance train and extended
            break
        else:
            count += 1
    return total_steps

def validate(model, args, total_steps):
    results = {}
    # Evaluate results
    results.update(validate_process(model, args, total_steps))
    wandb.log({
                "val_mace": results['val_mace'],
                "val_mace_conf_error": results['val_mace_conf_error']
            })
    return results['val_mace'], results['val_mace_conf_error']


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

    wandb.init(project="STGL-IHN", entity="xjh19971", config=vars(args))
        
    main(args)
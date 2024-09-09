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

from model.network import STHN
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
    model = STHN(args, for_training=True)
    logging.info(f"Parameter Count: {count_parameters(model.netG)}")

    if args.restore_ckpt is not None:
        model_med = torch.load(args.restore_ckpt, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=True)
        if args.two_stages and model_med['netG_fine'] is not None:
            model_med = torch.load(args.restore_ckpt, map_location='cuda:0')
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
            for key in list(model_med['netG_fine'].keys()):
                if key.startswith('module'):
                    del model_med['netG_fine'][key]
            model.netG_fine.load_state_dict(model_med['netG_fine'], strict=True)
        
    model.setup()
    model.netG.train()
    if args.two_stages:
        model.netG_fine.train()

    train_loader = datasets.fetch_dataloader(args, split="train")
    if os.path.exists(os.path.join(args.datasets_folder, args.dataset_name, "extended_queries.h5")):
        extended_loader = datasets.fetch_dataloader(args, split="extended")
    else:
        extended_loader = None

    total_steps = 0
    last_best_val_mace = None
    while total_steps <= args.num_steps:
        total_steps, last_best_val_mace = train(model, train_loader, args, total_steps, last_best_val_mace)
        if extended_loader is not None:
            total_steps, last_best_val_mace = train(model, extended_loader, args, total_steps, last_best_val_mace, train_step_limit=len(train_loader))

    test_dataset = datasets.fetch_dataloader(args, split='test')
    model_med = torch.load(args.save_dir + f'/{args.name}.pth')
    model.netG.load_state_dict(model_med['netG'], strict=False)
    if args.two_stages:
        model.netG_fine.load_state_dict(model_med['netG_fine'], strict=True)
    evaluate_SNet(model, test_dataset, batch_size=args.batch_size, args=args, wandb_log=True)

def train(model, train_loader, args, total_steps, last_best_val_mace, train_step_limit = None):
    count = 0
    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        tic = time.time()
        image1, image2, flow, _, query_utm, database_utm, _, _  = [x for x in data_blob]
        model.set_input(image1, image2, flow)
        metrics = model.optimize_parameters()
        if i_batch==0 and args.vis_all:
            save_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img1.png')
            save_img(torchvision.utils.make_grid(model.image_2, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img2.png')
            save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                            torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                            args.save_dir + '/train_overlap_gt.png')
            if not args.two_stages:
                save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                                torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0),
                                args.save_dir + f'/train_overlap_pred.png')
            else:
                save_img(torchvision.utils.make_grid(model.image_1_crop, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img1_crop.png')
                save_img(torchvision.utils.make_grid(model.image_2_crop, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img2_crop.png')
                save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                                torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0),
                                args.save_dir + f'/train_overlap_pred.png')
        model.update_learning_rate()
        if torch.isnan(model.loss_G):
            weights = model.optimizer_G.param_groups[0]['params']
            weights_flat = [torch.flatten(weight) for weight in weights]
            weights_1d = torch.cat(weights_flat)
            assert not torch.isnan(weights_1d).any()
            assert not torch.isinf(weights_1d).any()
            print(f"{weights_1d.max()}, {weights_1d.min()}")

            grad_flat = [torch.flatten(weight.grad) for weight in weights]
            grad_1d = torch.cat(grad_flat)
            assert not torch.isnan(grad_1d).any()
            assert not torch.isinf(grad_1d).any()
            print(f"{grad_1d.max()}, {grad_1d.min()}")
            raise KeyError("Detect NaN for loss")
        metrics["lr"] = model.scheduler_G.get_lr()
        toc = time.time()
        metrics['time'] = toc - tic
        wandb.log({
                "mace": metrics["mace"],
                "lr": metrics["lr"][0],
                "G_loss": metrics["G_loss"],
                "ce_loss": metrics["ce_loss"],
            },)
        total_steps += 1
        # Validate
        if total_steps % args.val_freq == args.val_freq - 1:
            current_val_mace = validate(model, args, total_steps)
            # plot_train(logger, args)
            # plot_val(logger, args)
            PATH = args.save_dir + f'/{total_steps+1}_{args.name}.pth'
            checkpoint = {
                "netG": model.netG.state_dict(),
                "netG_fine": model.netG_fine.state_dict() if args.two_stages else None,
            }
            torch.save(checkpoint, PATH)
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
    return total_steps, last_best_val_mace

def validate(model, args, total_steps):
    results = {}
    # Evaluate results
    results.update(validate_process(model, args, total_steps))
    wandb.log({
                "val_mace": results['val_mace'],
            })
    return results['val_mace']


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

    wandb.init(project="STHN", entity="xjh19971", config=vars(args))
        
    main(args)
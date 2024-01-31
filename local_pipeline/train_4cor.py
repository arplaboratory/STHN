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

from model.network import STHEGAN
from utils import count_parameters, Logger, save_img, save_overlap_img, setup_seed, Logger_, warp
import commons
from os.path import join
from evaluate import validate_process
import datasets_4cor_img as datasets
import parser
import wandb
import datetime
from uuid import uuid4

def main(args):
    args.save_dir = join(
    "logs",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
    )
    commons.setup_logging(args.save_dir)

    device = torch.device('cuda:'+ str(args.gpuid[0]))

    model = STHEGAN(args, for_training=True)
    model.setup()
    if args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
        model.netG.eval()
        for param in model.netG.parameters():
            param.requires_grad = False
    else:
        model.netG.train()
    print(f"Parameter Count: {count_parameters(model.netG)}")
    if args.use_ue:
        model.netD.train()
        print(f"Parameter Count: {count_parameters(model.netD)}")

    if args.restore_ckpt is not None:

        save_model = torch.load(args.restore_ckpt)
        
        model.netG.load_state_dict(save_model['netG'])
        if save_model['netD'] is not None:
            model.netD.load_state_dict(save_model['netD'])
        
    train_loader = datasets.fetch_dataloader(args, split="train")
    if os.path.exists(os.path.join(args.datasets_folder, args.dataset_name, "extended_queries.h5")):
        extended_loader = datasets.fetch_dataloader(args, split="extended")
    else:
        extended_loader = None
    logger = Logger(model, model.scheduler_G, args)

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, logger, args)
        if extended_loader is not None:
            train(model, extended_loader, logger, args, train_step_limit=len(train_loader))

def train(model, train_loader, logger, args, train_step_limit = None):
    count = 0
    last_best_val_mace = None
    last_best_val_mace_conf_error = None
    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        tic = time.time()
        image1, image2, flow, _, query_utm, database_utm  = [x.cuda() for x in data_blob]
        image2_w = warp(image2, flow)

        if i_batch==0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            save_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0), './watch/' + 'train_img1.bmp')
            save_img(torchvision.utils.make_grid(image2, nrow=16, padding = 16, pad_value=0), './watch/' + 'train_img2.bmp')
            save_img(torchvision.utils.make_grid(image2_w, nrow=16, padding = 16, pad_value=0), './watch/' + 'train_img2w.bmp')
            save_overlap_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0),
                             torchvision.utils.make_grid(image2_w, nrow=16, padding = 16, pad_value=0), 
                             './watch/' + 'train_overlap.png')

        model.set_input(image1, image2, flow)
        metrics = model.optimize_parameters()
        model.update_learning_rate()
        metrics["lr"] = model.scheduler_G.get_lr()
        toc = time.time()
        metrics['time'] = toc - tic
        logger.push(metrics)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:
            current_val_mace, current_val_mace_conf_error = validate(model, args, logger)
            # plot_train(logger, args)
            # plot_val(logger, args)
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            checkpoint = {
                "netG": model.netG.state_dict(),
                "netD": model.netD.state_dict() if args.use_ue else None,
            }
            torch.save(checkpoint, PATH)
            if args.use_ue and args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
                if last_best_val_mace_conf_error is None or last_best_val_mace_conf_error > current_val_mace_conf_error:
                    last_best_val_mace_conf_error = current_val_mace_conf_error
                    PATH = args.output + f'/{args.name}.pth'
                    torch.save(checkpoint, PATH)
            else:
                if last_best_val_mace is None or last_best_val_mace > current_val_mace:
                    last_best_val_mace = current_val_mace
                    PATH = args.output + f'/{args.name}.pth'
                    torch.save(checkpoint, PATH)

        if logger.total_steps >= args.num_steps:
            break
        
        if train_step_limit is not None and count >= train_step_limit: # Balance train and extended
            break
        else:
            count += 1

def validate(model, args, logger):
    results = {}
    # Evaluate results
    results.update(validate_process(model, args, logger))
    wandb.log({
                "step": logger.total_steps,
                "val_mace": results['val_mace'],
                "val_mace_conf_error": results['val_mace_conf_error']
            },)
    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])
    logger.val_steps_list.append(logger.total_steps)
    return results['val_mace'], results['val_mace_conf_error']


if __name__ == "__main__":
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join(
    "logs",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
    )
    commons.setup_logging(args.save_dir)
    setup_seed(0)

    wandb.init(project="STGL-IHN", entity="xjh19971", config=vars(args))
    sys.stdout = Logger_(args.logname, sys.stdout)
        
    main(args)
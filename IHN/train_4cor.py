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

from network import IHN
from utils import *

from evaluate import validate_process
import datasets_4cor_img as datasets

import wandb

def main(args):
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    model = IHN(args)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.train()
    print(f"Parameter Count: {count_parameters(model)}")
    optimizer, scheduler = fetch_optimizer(args, model)

    if args.restore_ckpt is not None:

        save_model = torch.load(args.restore_ckpt)
        model.load_state_dict(save_model['net'])

        optimizer.load_state_dict(save_model['optimizer'])

        scheduler.load_state_dict(save_model['scheduler'])
        
    train_loader = datasets.fetch_dataloader(args, split="train")
    if os.path.exists(os.path.join(args.datasets_folder, args.dataset_name, "extended_database.h5")):
        extended_loader = datasets.fetch_dataloader(args, split="extended")
    else:
        extended_loader = None
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if extended_loader is not None:
            train(model, extended_loader, optimizer, scheduler, logger, scaler, args, train_step_limit=len(train_loader))

    PATH = args.output + f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


def train(model, train_loader, optimizer, scheduler, logger, scaler, args, train_step_limit = None):
    count = 0
    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        tic = time.time()
        image1, image2, flow,  H, query_utm, database_utm  = [x.cuda() for x in data_blob]
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

        optimizer.zero_grad()

        four_pred = model(image1, image2, iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1)

        loss, metrics = sequence_loss(four_pred, flow, H, args.gamma, args) 
        metrics["lr"] = scheduler.get_lr()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()

        metrics['time'] = toc - tic
        logger.push(metrics)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:
            validate(model, args, logger)
            # plot_train(logger, args)
            # plot_val(logger, args)
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
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
    results.update(validate_process(model, args))
    wandb.log({
                "step": logger.total_steps,
                "val_mace": results['val_mace']
            },)
    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])
    logger.val_steps_list.append(logger.total_steps)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='RHWF', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    
    parser.add_argument('--gpuid', type=int, nargs='+', default = [0])
    parser.add_argument('--output', type=str, default='IHN_results/satellite', help='output directory to save checkpoints and plots')
    parser.add_argument('--logname', type=str, default='satellite.log', help='printing frequency')

    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true', help='warp once')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)
    
    parser.add_argument('--val_freq', type=int, default=10000, help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='printing frequency')

    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    
    parser.add_argument('--model_name', default='', help='specify model name')
    parser.add_argument('--resume', default=False, action='store_true', help='resume_training')
    
    parser.add_argument('--weight', action='store_true')
    parser.add_argument(
        "--datasets_folder", type=str, default="datasets", help="Path with all datasets"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="satellite_0_satellite_0_dense",
        help="Relative path of the dataset",
    )
    parser.add_argument(
        "--prior_location_threshold",
        type=int,
        default=-1,
        help="The threshold of search region from prior knowledge for train and test. If -1, then no prior knowledge"
    )
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50, help="_")
    parser.add_argument(
        "--G_contrast",
        type=str,
        default="none",
        choices=["none", "manual", "autocontrast", "equalize"],
        help="G_contrast"
    )
    parser.add_argument(
        "--output_norm",
        type=float,
        default=-1,
        help="Normalization for output"
    )
    args = parser.parse_args()

    setup_seed(1024)

    wandb.init(project="STGL-IHN", entity="xjh19971", config=vars(args))
    sys.stdout = Logger_(args.logname, sys.stdout)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    main(args)
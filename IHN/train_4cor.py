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

from network import IHN, STHEGAN
from utils import count_parameters, Logger, save_img, save_overlap_img, setup_seed, Logger_, warp

from evaluate import validate_process
import datasets_4cor_img as datasets

import wandb

def main(args):
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

    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_steps', type=int, default=200000)
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
        "--use_ue",
        action="store_true",
        help="train uncertainty estimator with GAN"
    )
    parser.add_argument(
        "--G_loss_lambda",
        type=float,
        default=1.0,
        help="G_loss_lambda only for homo"
    )
    parser.add_argument(
        "--D_net",
        type=str,
        default="patchGAN_deep",
        choices=["none", "patchGAN", "patchGAN_deep"],
        help="D_net"
    )
    parser.add_argument(
        "--GAN_mode",
        type=str,
        default="macegan",
        choices=["vanilla", "lsgan", "macegan", "macegancross"],
        help="Choices of GAN loss"
    )
    parser.add_argument("--device", type=str,
        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--train_ue_method",
        type=str,
        choices=['train_end_to_end', 'train_only_ue', 'train_only_ue_raw_input'],
        default='train_end_to_end',
        help="train uncertainty estimator"
    )
    parser.add_argument(
        "--ue_alpha",
        type=float,
        default=-0.1,
        help="Alpha for ue"
    )
    parser.add_argument(
        "--permute",
        action="store_true",
        help="Permute input images"
    )
    # parser.add_argument(
    #     "--resize_small",
    #     action="store_true",
    #     help="resize from 512 to 256"
    # )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=10
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        choices=['target', 'raw', 'target_raw'],
        default='target_raw',
        help="sample noise"
    )
    args = parser.parse_args()
    args.resize_small = True
    setup_seed(0)

    wandb.init(project="STGL-IHN", entity="xjh19971", config=vars(args))
    sys.stdout = Logger_(args.logname, sys.stdout)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    main(args)
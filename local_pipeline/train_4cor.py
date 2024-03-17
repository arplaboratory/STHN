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
    model.setup()
    if args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
        model.netG.eval()
        for param in model.netG.parameters():
            param.requires_grad = False
        if args.two_stages:
            model.netG_fine.eval()
            for param in model.netG_fine.parameters():
                param.requires_grad = False
    else:
        if args.restore_ckpt is None or args.finetune:
            model.netG.train()
        else:
            model.netG.eval()
        if args.two_stages:
            model.netG_fine.train()
    logging.info(f"Parameter Count: {count_parameters(model.netG)}")
    if args.use_ue and args.D_net != "ue_branch":
        model.netD.train()
        logging.info(f"Parameter Count: {count_parameters(model.netD)}")

    if args.restore_ckpt is not None:

        save_model = torch.load(args.restore_ckpt)
        
        model.netG.load_state_dict(save_model['netG'])
        if save_model['netD'] is not None:
            model.netD.load_state_dict(save_model['netD'])
        if save_model['netG_fine'] is not None:
            model.netG_fine.load_state_dict(save_model['netG_fine'])
        
    train_loader = datasets.fetch_dataloader(args, split="train")
    if os.path.exists(os.path.join(args.datasets_folder, args.dataset_name, "extended_queries.h5")):
        extended_loader = datasets.fetch_dataloader(args, split="extended")
    else:
        extended_loader = None

    total_steps = 0
    last_best_val_mace = None
    last_best_val_mace_conf_error = None
    while total_steps <= args.num_steps:
        total_steps, last_best_val_mace, last_best_val_mace_conf_error = train(model, train_loader, args, total_steps, last_best_val_mace, last_best_val_mace_conf_error)
        if extended_loader is not None:
            total_steps, last_best_val_mace, last_best_val_mace_conf_error = train(model, extended_loader, args, total_steps, last_best_val_mace, last_best_val_mace_conf_error, train_step_limit=len(train_loader))

    test_dataset = datasets.fetch_dataloader(args, split='test')
    model_med = torch.load(args.save_dir + f'/{args.name}.pth')
    model.netG.load_state_dict(model_med['netG'])
    if args.use_ue and args.D_net != "ue_branch":
        model.netD.load_state_dict(model_med['netD'])
    if args.two_stages:
        model.netG_fine.load_state_dict(model_med['netG_fine'])
    evaluate_SNet(model, test_dataset, batch_size=args.batch_size, args=args, wandb_log=True)

def train(model, train_loader, args, total_steps, last_best_val_mace, last_best_val_mace_conf_error, train_step_limit = None):
    count = 0
    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        tic = time.time()
        image1, image2, flow, _, query_utm, database_utm, image1_ori  = [x for x in data_blob]
        model.set_input(image1, image2, flow, image1_ori)
        if i_batch==0:
            save_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img1.png')
            save_img(torchvision.utils.make_grid(image2, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img2.png')
            save_overlap_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0),
                             torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                             args.save_dir + '/train_overlap_gt.png')
        metrics = model.optimize_parameters()
        if i_batch==0 and args.train_ue_method != 'train_only_ue_raw_input':
            save_overlap_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0),
                            torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0),
                            args.save_dir + f'/train_overlap_pred.png')
            if args.two_stages:
                save_img(torchvision.utils.make_grid(model.image_1_crop, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img1_crop.png')
                save_overlap_img(torchvision.utils.make_grid(model.image_1_crop, nrow=16, padding = 16, pad_value=0),
                                torchvision.utils.make_grid(model.image_2, nrow=16, padding = 16, pad_value=0), 
                                args.save_dir + '/train_overlap_crop.png')
        model.update_learning_rate()
        if args.train_ue_method != 'train_only_ue_raw_input':
            metrics["lr"] = model.scheduler_G.get_lr()
        else:
            metrics["lr"] = model.scheduler_D.get_lr()
        toc = time.time()
        metrics['time'] = toc - tic
        wandb.log({
                "mace": metrics["mace"] if args.train_ue_method == 'train_end_to_end' else 0,
                "lr": metrics["lr"],
                "G_loss": metrics["G_loss"] if args.train_ue_method == 'train_end_to_end' else 0,
                "GAN_loss": metrics["GAN_loss"] if args.train_ue_method == 'train_end_to_end' and args.use_ue else 0,
                "D_loss": metrics["D_loss"] if args.use_ue and args.D_net != "ue_branch" else 0,
                "ue_loss": metrics["ue_loss"] if args.use_ue and args.D_net == "ue_branch" else 0
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
                "netD": model.netD.state_dict() if args.use_ue and args.D_net != "ue_branch" else None,
                "netG_fine": model.netG_fine.state_dict() if args.two_stages else None,
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
    return total_steps, last_best_val_mace, last_best_val_mace_conf_error

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

    wandb.init(project="STHN", entity="xjh19971", config=vars(args))
        
    main(args)
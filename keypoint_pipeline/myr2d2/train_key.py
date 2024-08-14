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

from model.network import KeyNet
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
    model = KeyNet(args, for_training=True)
    logging.info(f"Parameter Count: {count_parameters(model.netG)}")
        
    model.setup()
    model.netG.train()

    train_loader = datasets.fetch_dataloader(args, split="train")
    if os.path.exists(os.path.join(args.datasets_folder, args.dataset_name, "extended_queries.h5")):
        extended_loader = datasets.fetch_dataloader(args, split="extended")
    else:
        extended_loader = None

    total_steps = 0
    last_best_val_loss = None
    while total_steps <= args.num_steps:
        total_steps, last_best_val_loss = train(model, train_loader, args, total_steps, last_best_val_loss)
        if extended_loader is not None:
            total_steps, last_best_val_loss = train(model, extended_loader, args, total_steps, last_best_val_loss, train_step_limit=len(train_loader))

    test_dataset = datasets.fetch_dataloader(args, split='test')
    model_med = torch.load(args.save_dir + f'/{args.name}.pth')
    model.netG.load_state_dict(model_med['netG'], strict=True)
    evaluate_SNet(model, test_dataset, batch_size=args.batch_size, args=args, wandb_log=True)

def train(model, train_loader, args, total_steps, last_best_val_loss, train_step_limit = None):
    count = 0
    for i_batch, data_blob in enumerate(tqdm(train_loader)):
        tic = time.time()
        image1, image2, flow, _, query_utm, database_utm, _, _  = [x for x in data_blob]
        model.set_input(image1, image2, flow)
        metrics = model.optimize_parameters()
        # if i_batch==0:
        #     save_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img1.png')
        #     save_img(torchvision.utils.make_grid(model.image_2, nrow=16, padding = 16, pad_value=0), args.save_dir + '/train_img2.png')
        #     save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
        #                      torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
        #                      args.save_dir + '/train_overlap_gt.png')
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
                "lr": metrics["lr"][0],
                "R_loss": metrics["loss_reliability"] if not args.disable_reliability else metrics["loss_pixAP"],
                "C_loss": metrics["loss_cosim16"],
                "P_loss": metrics["loss_peaky16"],
            },)
        total_steps += 1
        # Validate
        if total_steps % args.val_freq == args.val_freq - 1:
            current_val_loss = validate(model, args, total_steps)
            # plot_train(logger, args)
            # plot_val(logger, args)
            PATH = args.save_dir + f'/{total_steps+1}_{args.name}.pth'
            checkpoint = {
                "netG": model.netG.state_dict()
            }
            torch.save(checkpoint, PATH)
            if last_best_val_loss is None or last_best_val_loss > current_val_loss:
                logging.info(f"Saving best model, last_best_val_loss: {last_best_val_loss}, current_val_loss: {current_val_loss}")
                last_best_val_loss = current_val_loss
                PATH = args.save_dir + f'/{args.name}.pth'
                torch.save(checkpoint, PATH)
            else:
                logging.info(f"No Saving, last_best_val_loss: {last_best_val_loss}, current_val_loss: {current_val_loss}")

        if total_steps >= args.num_steps:
            break
        
        if train_step_limit is not None and count >= train_step_limit: # Balance train and extended
            break
        else:
            count += 1
    return total_steps, last_best_val_loss

def validate(model, args, total_steps):
    results = {}
    # Evaluate results
    results.update(validate_process(model, args, total_steps))
    wandb.log({
                "val_loss": results['val_loss'],
            })
    return results['val_loss']


if __name__ == "__main__":
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join(
    "logs",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}",
    )
    commons.setup_logging(args.save_dir, console='info')
    setup_seed(args.seed)

    wandb.init(project="UAGL", entity="xjh19971", config=vars(args))
        
    main(args)
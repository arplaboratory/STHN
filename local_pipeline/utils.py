import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from skimage import io
import random
import sys
import torch.optim as optim
from PIL import Image
import logging
import wandb
import matplotlib.pyplot as plt
from datasets_4cor_img import inv_base_transforms


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def save_img(img, path):
    img = inv_base_transforms(img.detach().cpu())
    img.save(path)


def save_overlap_img(img1, img2, path):
    img1 = inv_base_transforms(img1.detach().cpu())
    img1 = np.array(img1)
    img2 = inv_base_transforms(img2.detach().cpu())
    img2 = np.array(img2)
    plt.figure(figsize=(50, 10), dpi=200)
    plt.axis('off')
    plt.imshow(img2)
    plt.imshow(img1, alpha=0.25)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask


def sequence_loss(four_preds, flow_gt, gamma, args, metrics):
    """ Loss function defined over sequence of flow predictions """

    flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2)).to(four_preds[0].device)
    flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
    flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
    flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
    flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

    ce_loss = 0.0

    for i in range(args.iters_lev0):
        i_weight = gamma ** (args.iters_lev0 - i - 1)
        i4cor_loss = (four_preds[i] - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()

    if args.two_stages: # Assume two_stages and iters_lev0 == iters_lev1
        for i in range(args.iters_lev0, args.iters_lev0 + args.iters_lev0):
            i_weight = gamma ** (args.iters_lev0 + args.iters_lev0 - i - 1)
            i4cor_loss = (four_preds[i] - flow_4cor).abs()
            ce_loss += i_weight * (i4cor_loss).mean()

    mace = torch.sum((four_preds[-1] - flow_4cor) ** 2, dim=1).sqrt()
    metrics['1px'] = (mace < 1).float().mean().item()
    metrics['3px'] = (mace < 3).float().mean().item()
    metrics['mace'] = mace.mean().item()
    return ce_loss, metrics


def fetch_optimizer(args, model_para):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model_para, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler
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

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

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
    npimg = img.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    img = Image.fromarray(npimg)
    img.save(path)

def save_overlap_img(img1, img2, path):
    npimg = img1.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    img1 = npimg.astype(np.uint8)
    npimg = img2.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    img2 = npimg.astype(np.uint8)
    plt.figure(dpi=200)
    plt.axis('off')
    plt.imshow(img2)
    plt.imshow(img1, alpha=0.35)
    plt.savefig(path, bbox_inches='tight')
    
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

class Logger_(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])
        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)
        # logging running loss to total loss
        self.train_mace_list.append(np.mean(self.running_loss_dict['mace']))
        self.train_steps_list.append(self.total_steps)
        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(metrics[key])
        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            wandb.log({
                "step": self.total_steps,
                "mace": np.mean(self.running_loss_dict['mace']),
                "lr": np.mean(self.running_loss_dict['lr'])
            },)
            self._print_training_status()
            self.running_loss_dict = {}
            
def sequence_loss(four_preds, flow_gt, H, gamma, args):
    """ Loss function defined over sequence of flow predictions """

    flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2)).to(four_preds[0].device)
    flow_4cor[:,:, 0, 0]  = flow_gt[:,:, 0, 0]
    flow_4cor[:,:, 0, 1] = flow_gt[:,:,  0, -1]
    flow_4cor[:,:, 1, 0] = flow_gt[:,:, -1, 0]
    flow_4cor[:,:, 1, 1] = flow_gt[:,:, -1, -1]

    ce_loss = 0.0

    for i in range(args.iters_lev0):
        i_weight = gamma**(args.iters_lev0 - i - 1)
        i4cor_loss = (four_preds[i] - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()

    if args.lev1:
        for i in range(args.iters_lev0, args.iters_lev1 + args.iters_lev0):
            i_weight = gamma ** (args.iters_lev1 + args.iters_lev0 - i - 1)
            i4cor_loss = (four_preds[i] - flow_4cor).abs()
            ce_loss += i_weight * (i4cor_loss).mean()

    mace = torch.sum((four_preds[-1] - flow_4cor)**2, dim=1).sqrt()
    metrics = {
        '1px': (mace < 1).float().mean().item(),
        '3px': (mace < 3).float().mean().item(),
        'mace': mace.mean().item(),
    }
    return ce_loss , metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler
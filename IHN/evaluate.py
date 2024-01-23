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

@torch.no_grad()
def validate_process(model, args, logger):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.netG.eval()
    if args.use_ue:
        model.netD.eval()
    mace_list = []
    mace_conf_list = []
    mace_conf_error_list = []
    val_loader = datasets.fetch_dataloader(args, split='val')
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        image1, image2, flow_gt,  H, _, _  = [x.to(model.netG.module.device) for x in data_blob]
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        image1 = image1.to(model.netG.module.device)
        image2 = image2.to(model.netG.module.device)
        model.set_input(image1, image2, flow_gt)
        model.forward()
        if i_batch == 0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            # Visualize
            save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                            torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                            './watch/' + 'train_overlap_pred.png')
            save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                            torchvision.utils.make_grid(model.real_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                            './watch/' + 'train_overlap_gt.png')
        four_pr = model.four_pred
        mace = torch.sum((four_pr.cpu().detach() - flow_4cor) ** 2, dim=1).sqrt()
        mace_list.append(mace.view(-1).numpy())
        if args.use_ue:
            mace_gt = (flow_4cor - flow_4cor)**2
            mace_gt = ((mace_gt[:,0,:,:] + mace_gt[:,1,:,:])**0.5)
            mace_gt_vec = torch.mean(torch.mean(mace_gt, dim=1), dim=1)
            mace_pred = (flow_4cor - four_pr.cpu().detach())**2
            mace_pred = ((mace_pred[:,0,:,:] + mace_pred[:,1,:,:])**0.5)
            mace_pred_vec = torch.mean(torch.mean(mace_pred, dim=1), dim=1)
            conf_pred, conf_gt = model.predict_uncertainty(GAN_mode=args.GAN_mode)
            conf_pred_vec = torch.mean(conf_pred, dim=[1, 2, 3])
            conf_gt_vec = torch.mean(conf_gt, dim=[1, 2, 3])
            mace = torch.sum((four_pr.cpu().detach() - flow_4cor) ** 2, dim=1).sqrt()
            mace_list.append(mace.view(-1).numpy())
            mace_conf_error = F.mse_loss(conf_pred_vec.cpu(), torch.exp(args.ue_alpha * torch.mean(torch.mean(mace, dim=1), dim=1)))
            mace_conf_error_list.append(mace_conf_error.numpy())
            for i in range(len(mace_pred_vec)):
                mace_conf_list.append((mace_pred_vec[i].item(), conf_pred_vec[i].item(), mace_gt_vec[i].item(), conf_gt_vec[i].item()))

    if args.train_only_ue:
        model.netG.eval()
    else:
        model.netG.train()
    if args.use_ue:
        model.netD.train()
        mace_conf_list = np.array(mace_conf_list)
        # plot mace conf
        plt.figure()
        # plt.axis('equal')
        plt.scatter(mace_conf_list[:,0], mace_conf_list[:,1], s=5)
        plt.scatter(mace_conf_list[:,2], mace_conf_list[:,3], s=5)
        plt.xlabel("MACE")
        plt.ylabel("conf")
        plt.savefig(args.output + f'/{logger.total_steps}_conf.png')
        plt.close()
    mace = np.mean(np.concatenate(mace_list))
    mace_conf_error = np.mean(np.array(mace_conf_error_list)) if args.use_ue else 0
    print("Validation MACE: %f" % mace)
    print("Validation MACE CONF ERROR: %f" % mace_conf_error)
    return {'val_mace': mace, 'mace_conf_error': mace_conf_error}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()
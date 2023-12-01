import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

import datasets_4cor_img as datasets
from utils import *

@torch.no_grad()
def validate_process(model, args, logger):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.netG.eval()
    if args.use_ue:
        model.netD.eval()
    mace_list = []
    mace_conf_list = []
    args.batch_size = 16
    val_dataset = datasets.fetch_dataloader(args, split='val')
    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        image1, image2, flow_gt,  H, _, _  = [x.to(model.netG.module.device) for x in data_blob]
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        # if i_batch == 0:
        #     if not os.path.exists('watch'):
        #         os.makedirs('watch')
        #     save_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=0),
        #             './watch/' + "b1_epoch_" + str(i_batch).zfill(5) + "_iter_" + '.bmp')
        #     save_img(torchvision.utils.make_grid(image2, nrow=16, padding = 16, pad_value=0),
        #             './watch/' + "b2_epoch_" + str(i_batch).zfill(5) + "_iter_" + '.bmp')

        image1 = image1.to(model.netG.module.device)
        image2 = image2.to(model.netG.module.device)
        model.set_input(image1, image2, flow_gt)
        model.forward()
        four_pr = model.four_pred
        mace = torch.sum((four_pr.cpu().detach() - flow_4cor) ** 2, dim=0).sqrt()
        mace_list.append(mace.view(-1).numpy())
        if args.use_ue:
            mace_ = (flow_4cor - four_pr.cpu().detach())**2
            mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
            mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
            conf = model.predict_uncertainty()
            conf_vec = torch.mean(conf, dim=[1, 2, 3])
            for i in range(len(mace_vec)):
                mace_conf_list.append((mace_vec[i].item(), conf_vec[i].item()))

    model.netG.train()
    if args.use_ue:
        model.netD.train()
        mace_conf_list = np.array(mace_conf_list)
        # plot mace conf
        plt.figure()
        plt.scatter(mace_conf_list[:,0], mace_conf_list[:,1], s=5)
        plt.xlabel("MACE")
        plt.ylabel("conf")
        plt.savefig('/'.join(args.output.split('/')[:-1]) + f'/{logger.total_steps}_conf.png')
        plt.close()
    mace = np.mean(np.concatenate(mace_list))
    print("Validation MACE: %f" % mace)
    return {'val_mace': mace}

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
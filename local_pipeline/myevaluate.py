import numpy as np
import os
import torch
import argparse
from model.network import STHEGAN
from utils import save_overlap_img, save_img, setup_seed
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
from tqdm import tqdm
import cv2
import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
from plot_hist import plot_hist_helper
import torch.nn.functional as F
import parser
from datetime import datetime
from os.path import join
import commons
import logging

def test(args):
    model = STHEGAN(args)
    model_med = torch.load(args.eval_model, map_location='cuda:0')

    for key in list(model_med['netG'].keys()):
        model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
    for key in list(model_med['netG'].keys()):
        if key.startswith('module'):
            del model_med['netG'][key]
    model.netG.load_state_dict(model_med['netG'])
    if args.use_ue:
        for key in list(model_med['netD'].keys()):
            model_med['netD'][key.replace('module.','')] = model_med['netD'][key]
        for key in list(model_med['netD'].keys()):
            if key.startswith('module'):
                del model_med['netD'][key]
        model.netD.load_state_dict(model_med['netD'])
    
    model.setup() 
    model.netG.eval()
    if args.use_ue:
        model.netD.eval()

    val_dataset = datasets.fetch_dataloader(args, split='test')
    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args)
    
def evaluate_SNet(model, val_dataset, batch_size=0, args = None):

    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)
    total_flow = torch.empty(0)
    total_mace_conf_error = torch.empty(0)
    timeall=[]
    mace_conf_list = []
    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        img1, img2, flow_gt,  H, query_utm, database_utm  = [x.to(model.device) for x in data_blob]

        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")

        if i_batch%100 == 0:
            save_img(torchvision.utils.make_grid((img1)),
                     args.save_dir + "/b1_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.bmp')
            save_img(torchvision.utils.make_grid((img2)),
                     args.save_dir + "/b2_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.bmp')

        time_start = time.time()
        model.set_input(img1, img2, flow_gt)
        model.forward(use_raw_input=(args.train_ue_method == 'train_only_ue_raw_input'), noise_std=args.noise_std)
        four_pred = model.four_pred
        time_end = time.time()
        timeall.append(time_end-time_start)
        # print(time_end-time_start)

        flow_4cor = torch.zeros((four_pred.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        mace_ = (flow_4cor - four_pred.cpu().detach())**2
        mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
        mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
      
        flow_ = (flow_4cor)**2
        flow_ = ((flow_[:,0,:,:] + flow_[:,1,:,:])**0.5)
        flow_vec = torch.mean(torch.mean(flow_, dim=1), dim=1)
        total_mace = torch.cat([total_mace,mace_vec], dim=0)
        final_mace = torch.mean(total_mace).item()
        total_flow = torch.cat([total_flow,flow_vec], dim=0)
        final_flow = torch.mean(total_flow).item()
        
        if args.use_ue:
            with torch.no_grad():
                conf_pred, conf_gt = model.predict_uncertainty(GAN_mode=args.GAN_mode)
            conf_vec = torch.mean(conf_pred, dim=[1, 2, 3])
            conf_gt_vec = torch.mean(conf_gt, dim=[1,2,3])
            logging.debug(f"conf_pred_diff:{conf_vec.cpu() - torch.exp(args.ue_alpha * mace_vec)}.\n conf_gt:{conf_gt_vec.cpu()}.")
            logging.debug(f"pred_mace:{mace_vec}")
            mace_conf_error_vec = F.l1_loss(conf_vec.cpu(), torch.exp(args.ue_alpha * mace_vec))
            total_mace_conf_error = torch.cat([total_mace_conf_error, mace_conf_error_vec.reshape(1)], dim=0)
            final_mace_conf_error = torch.mean(total_mace_conf_error).item()
            for i in range(len(mace_vec)):
                mace_conf_list.append((mace_vec[i].item(), conf_vec[i].item()))

        if i_batch%10000 == 0:
            save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                            torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                            args.save_dir + f'/eval_overlap_{i_batch}_{mace_vec.mean().item()}.png')
    logging.info("MACE Metric: ", final_mace)
    if args.use_ue:
        mace_conf_list = np.array(mace_conf_list)
        # plot mace conf
        plt.figure()
        # plt.axis('equal')
        plt.scatter(mace_conf_list[:,0], mace_conf_list[:,1], s=1)
        x = np.linspace(0, 100, 400)
        y = np.exp(args.ue_alpha * x)
        plt.plot(x, y, label='f(x) = exp(-0.1x)', color='red')
        plt.legend()
        plt.savefig(args.save_dir + f'/final_conf.png')
        plt.close()
        plt.figure()
        n, bins, patches = plt.hist(x=mace_conf_list[:,1], bins=np.linspace(0, 1, 20))
        logging.info(n)
        plt.close()
        logging.info("MACE CONF ERROR Metric: ", final_mace_conf_error)
    logging.info(np.mean(np.array(timeall[1:-1])))
    io.savemat(args.save_dir + '/resmat', {'matrix': total_mace.numpy()})
    np.save(args.save_dir + '/resnpy.npy', total_mace.numpy())
    io.savemat(args.save_dir + '/flowmat', {'matrix': total_flow.numpy()})
    np.save(args.save_dir + '/flownpy.npy', total_flow.numpy())
    plot_hist_helper(args.save_dir)

if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join(
    "test",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    commons.setup_logging(args.save_dir)
    setup_seed(0)
    
    test(args)
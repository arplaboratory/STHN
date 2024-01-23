import numpy as np
import os
import torch
import argparse
from network import STHEGAN
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

setup_seed(0)
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
            print("Check the reproducibility by UTM:")
            print(f"the first 5th query UTMs: {query_utm[:5]}")
            print(f"the first 5th database UTMs: {database_utm[:5]}")

        if i_batch%1000 == 0:
            save_img(torchvision.utils.make_grid((img1)),
                     '/'.join(args.model.split('/')[:-1]) + "/b1_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.bmp')
            save_img(torchvision.utils.make_grid((img2)),
                     '/'.join(args.model.split('/')[:-1]) + "/b2_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.bmp')

        time_start = time.time()
        model.set_input(img1, img2, flow_gt)
        model.forward()
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
        # print(mace_.mean())
        # print("MACE Metric: ", final_mace)
        
        if args.use_ue:
            with torch.no_grad():
                conf_pred, conf_gt = model.predict_uncertainty(GAN_mode=args.GAN_mode)
            print(f"conf_pred:{torch.mean(conf_pred, dim=[1,2,3])}.\n conf_gt:{torch.mean(conf_gt, dim=[1,2,3])}.")
            print(f"pred_mace:{mace_vec}")
            conf_vec = torch.mean(conf_pred, dim=[1, 2, 3])
            mace_conf_error_vec = F.mse_loss(conf_vec.cpu(), torch.exp(args.ue_alpha * mace_vec))
            total_mace_conf_error = torch.cat([total_mace_conf_error, mace_conf_error_vec.reshape(1)], dim=0)
            final_mace_conf_error = torch.mean(total_mace_conf_error).item()
            for i in range(len(mace_vec)):
                mace_conf_list.append((mace_vec[i].item(), conf_vec[i].item()))

        if i_batch%1000 == 0:
            save_overlap_img(torchvision.utils.make_grid(model.image_1, nrow=16, padding = 16, pad_value=0),
                            torchvision.utils.make_grid(model.fake_warped_image_2, nrow=16, padding = 16, pad_value=0), 
                            '/'.join(args.model.split('/')[:-1]) + f'/eval_overlap_{i_batch}_{mace_vec.mean().item()}.png')
    print("MACE Metric: ", final_mace)
    if args.use_ue:
        mace_conf_list = np.array(mace_conf_list)
        # plot mace conf
        plt.figure()
        # plt.axis('equal')
        plt.scatter(mace_conf_list[:,0], mace_conf_list[:,1], s=5)
        x = np.linspace(0, 100, 400)
        y = np.exp(args.ue_alpha * x)
        plt.plot(x, y, label='f(x) = exp(-0.1x)', color='red')
        plt.legend()
        plt.savefig('/'.join(args.model.split('/')[:-1]) + f'/final_conf.png')
        plt.close()
        print("MACE CONF ERROR Metric: ", final_mace_conf_error)
    print(np.mean(np.array(timeall[1:-1])))
    io.savemat(f"{'/'.join(args.model.split('/')[:-1])}" + '/' + args.savemat, {'matrix': total_mace.numpy()})
    np.save(f"{'/'.join(args.model.split('/')[:-1])}" + '/' + args.savedict, total_mace.numpy())
    io.savemat(f"{'/'.join(args.model.split('/')[:-1])}" + '/' + args.savematflow, {'matrix': total_flow.numpy()})
    np.save(f"{'/'.join(args.model.split('/')[:-1])}" + '/' + args.savedictflow, total_flow.numpy())
    plot_hist_helper(f"{'/'.join(args.model.split('/')[:-1])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/IHN/IHN.pth',help="restore checkpoint")
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=3)
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--savemat', type=str,  default='resmat')
    parser.add_argument('--savedict', type=str, default='resnpy')
    parser.add_argument('--savematflow', type=str,  default='flowmat')
    parser.add_argument('--savedictflow', type=str, default='flownpy')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')    
    parser.add_argument('--lev0', default=False, action='store_true',
                        help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true',
                        help='warp once')
    parser.add_argument('--weight', default=False, action='store_true',
                        help='weight')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name_lev0', default='', help='specify model0 name')
    parser.add_argument('--model_name_lev1', default='', help='specify model0 name')
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
        "--D_net",
        type=str,
        default="patchGAN",
        choices=["none", "patchGAN", "patchGAN_deep"],
        help="D_net"
    )
    parser.add_argument(
        "--GAN_mode",
        type=str,
        default="vanilla",
        choices=["vanilla", "lsgan", "macegan"],
        help="Choices of GAN loss"
    )
    parser.add_argument(
        "--use_ue",
        action="store_true",
        help="train uncertainty estimator with GAN"
    )
    parser.add_argument("--device", type=str,
        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--ue_alpha",
        type=float,
        default=-0.1,
        help="Alpha for ue"
    )
    args = parser.parse_args()
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    model = STHEGAN(args)
    model_med = torch.load(args.model, map_location='cuda:0')

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

    val_dataset = datasets.fetch_dataloader(args, split='val')
    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args)
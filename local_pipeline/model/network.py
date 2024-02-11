import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox
from update import GMA
from extractor import BasicEncoderQuarter
from corr import CorrBlock
from utils import coords_grid, sequence_loss, fetch_optimizer, warp
import os
import sys
from model.pix2pix_networks.networks import GANLoss, NLayerDiscriminator
from model.sync_batchnorm import convert_model
import wandb
import torchvision
import random

autocast = torch.cuda.amp.autocast
class IHN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        if self.args.lev0:
            sz = 64
            self.update_block_4 = GMA(self.args, sz)
        if self.args.lev1:
            sz = 128
            self.update_block_2 = GMA(self.args, sz)

    def get_flow_now_4(self, four_point):
        four_point = four_point / 4
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def get_flow_now_2(self, four_point):
        four_point = four_point / 2
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4).to(img.device)
        coords1 = coords_grid(N, H//4, W//4).to(img.device)

        return coords0, coords1

    def initialize_flow_2(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//2, W//2).to(img.device)
        coords1 = coords_grid(N, H//2, W//2).to(img.device)

        return coords0, coords1

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=3):
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.args.mixed_precision):
            fmap1_64, fmap1_128 = self.fnet1(image1)
            fmap2_64, _ = self.fnet1(image2)
        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4)
        coords0, coords1 = self.initialize_flow_4(image1)
        sz = fmap1_64.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        four_point_predictions = []

        for itr in range(iters_lev0):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.weight:
                    delta_four_point, weight = self.update_block_4(corr, flow)
                else:
                    delta_four_point = self.update_block_4(corr, flow)
                    
            four_point_disp =  four_point_disp + delta_four_point
            four_point_predictions.append(four_point_disp)
            coords1 = self.get_flow_now_4(four_point_disp)


        if self.args.lev1:# next resolution
            four_point_disp_med = four_point_disp 
            flow_med = coords1 - coords0
            flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4    
            image2 = warp(image2, flow_med)
                      
            with autocast(enabled=self.args.mixed_precision):
                _, fmap2_128 = self.fnet1(image2)
            fmap1 = fmap1_128.float()
            fmap2 = fmap2_128.float()
            
            corr_fn = CorrBlock(fmap1, fmap2, num_levels = 2, radius= 4)
            coords0, coords1 = self.initialize_flow_2(image1)                
            sz = fmap1.shape
            self.sz = sz
            four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
            
            for itr in range(iters_lev1):
                corr = corr_fn(coords1)
                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    if self.args.weight:
                        delta_four_point, weight = self.update_block_2(corr, flow)
                    else:
                        delta_four_point = self.update_block_2(corr, flow)
                four_point_disp = four_point_disp + delta_four_point
                four_point_predictions.append(four_point_disp)            
                coords1 = self.get_flow_now_2(four_point_disp)            
            
            four_point_disp = four_point_disp + four_point_disp_med

        return four_point_predictions, four_point_disp


class STHEGAN():
    def __init__(self, args, for_training=False):
        super().__init__()
        self.args = args
        self.device = args.device
        self.four_point_org_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_single[:, :, 0, 1] = torch.Tensor([256 - 1, 0]).to(self.device)
        self.four_point_org_single[:, :, 1, 0] = torch.Tensor([0, 256 - 1]).to(self.device)
        self.four_point_org_single[:, :, 1, 1] = torch.Tensor([256 - 1, 256 - 1]).to(self.device)
        self.netG = IHN(args)
        if self.args.two_stages:
            self.netG_fine = IHN(args)
        if args.use_ue:
            if args.D_net == 'patchGAN':
                self.netD = NLayerDiscriminator(9, norm="instance") # satellite=3 thermal=3 warped_thermal=3. norm should be instance?
            elif args.D_net == 'patchGAN_deep':
                self.netD = NLayerDiscriminator(9, n_layers=4, norm="instance")
            else:
                raise NotImplementedError()
            self.criterionGAN = GANLoss(args.GAN_mode).to(args.device)
        self.criterionAUX = sequence_loss
        if for_training:
            self.optimizer_G, self.scheduler_G = fetch_optimizer(args, self.netG)
            if args.use_ue:
                self.optimizer_D, self.scheduler_D = fetch_optimizer(args, self.netD)
            self.G_loss_lambda = args.G_loss_lambda
            
    def setup(self):
        if hasattr(self, 'netD'):
            self.netD = self.init_net(self.netD)
        self.netG = self.init_net(self.netG)
        if hasattr(self, 'netG_fine'):
            self.netG_fine = self.init_net(self.netG_fine)

    def init_net(self, model):
        model = torch.nn.DataParallel(model)
        if torch.cuda.device_count() >= 2:
            # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
            model = convert_model(model)
            model = model.to(self.device)
        return model
    
    def set_input(self, A, B, flow_gt=None, B_ori=None):
        self.image_1 = A.to(self.device)
        self.image_2 = B.to(self.device)
        self.image_2_ori = B_ori.to(self.device)
        self.flow_gt = flow_gt
        if self.flow_gt is not None:
            self.flow_4cor = torch.zeros((self.flow_gt.shape[0], 2, 2, 2)).to(self.device)
            self.flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            self.flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            self.flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            self.flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            self.real_warped_image_2 = mywarp(self.image_2, self.flow_gt, self.four_point_org_single)
        else:
            self.real_warped_image_2 = None

    def predict_uncertainty(self, GAN_mode='vanilla'):
        fake_AB = torch.cat((self.image_1, self.image_2, self.fake_warped_image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB_conf = self.netD(fake_AB)
        if GAN_mode == 'vanilla':
            fake_AB_conf = nn.Sigmoid()(fake_AB_conf)
        if self.real_warped_image_2 is not None:
            real_AB = torch.cat((self.image_1, self.image_2, self.real_warped_image_2), 1)
            real_AB_conf = self.netD(real_AB)
            if GAN_mode == 'vanilla':
                real_AB_conf = nn.Sigmoid()(real_AB_conf)
        else:
            real_AB_conf = None
        return fake_AB_conf, real_AB_conf
        
    def forward(self, use_raw_input=False, noise_std=0, sample_method="target_raw"):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not use_raw_input:
            self.four_preds_list, self.four_pred = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, iters_lev1=self.args.iters_lev1)
            if self.args.two_stages:
                image_1_crop, bbox_s, resize_ratio = self.get_cropped_satellite_image(self.image_1, self.four_pred, self.args.fine_padding)
                image_2_crop = self.get_aligned_thermal_image(self.image_2, self.four_pred, bbox_s)
                self.four_preds_list_fine, self.four_pred_fine = self.netG_fine(image1=image_1_crop, image2=image_2_crop, iters_lev0=self.args.iters_lev0, iters_lev1=self.args.iters_lev1)
                self.four_preds_list, self.four_pred = self.combine_coarse_fine(self.four_preds_list, self.four_pred, self.four_preds_list_fine, self.four_pred_fine, resize_ratio)
        else:
            if sample_method == "target":
                self.four_pred = self.flow_4cor + noise_std * torch.randn(self.flow_4cor.shape[0], 2, 2, 2).to(self.device)
            elif sample_method == "raw":
                self.four_pred = torch.zeros_like(self.flow_4cor) + noise_std * torch.randn(self.flow_4cor.shape[0], 2, 2, 2).to(self.device)
            elif sample_method == "target_raw":
                if random.random() > 0.5:
                    self.four_pred = self.flow_4cor + noise_std * torch.randn(self.flow_4cor.shape[0], 2, 2, 2).to(self.device)
                else:
                    self.four_pred = torch.zeros_like(self.flow_4cor) + noise_std * torch.randn(self.flow_4cor.shape[0], 2, 2, 2).to(self.device)
            else:
                raise NotImplementedError()
        self.fake_warped_image_2 = mywarp(self.image_2, self.four_pred, self.four_point_org_single)

    def get_cropped_satellite_image(self, image_1, four_pred, fine_padding):
        x = four_pred[:, 0, :, :] # B, H, W
        y = four_pred[:, 1, :, :] # B, H, W
        left = torch.min(x, dim=[1, 2])  # B
        right = torch.max(x, dim=[1, 2]) # B
        top = torch.min(y, dim=[1, 2]) # B
        bottom = torch.max(y, dim=[1, 2]) # B
        w = torch.max((torch.stack([right-left, bottom-top], dim=1)), dim=1) # B
        c = torch.stack([(left + right)/2, (bottom + top)/2]) # B, 2
        w_padded = w + 2 * fine_padding
        crop_top_left = c - w_padded / 2 # B, 2 = x, y
        bbox_s = bbox.bbox_generator(crop_top_left[:, 0], crop_top_left[:, 1], w_padded, w_padded)
        resize_ratio = w_padded / 256
        image_1_crop = tgm.crop_and_resize(image_1, bbox_s, (256, 256))
        return image_1_crop, bbox_s, resize_ratio
        
    def get_aligned_thermal_image(self, image_2, four_pred, bbox_s):
        four_cor_bbox = four_pred.permute(0, 2, 1).view(four_cor_bbox.shape[0], 2, 2, 2) # B, 4, 2 to B, 2, 2, 2
        four_pred_crop = four_pred - four_cor_bbox # set to align with cropped satellite images
        image_2_crop = mywarp(image_2, four_pred_crop, self.four_point_org_single)
        return image_2_crop
    
    def combine_coarse_fine(four_preds_list, four_pred, four_preds_list_fine, four_pred_fine, resize_ratio):
        four_preds_list_fine = four_preds_list_fine * resize_ratio + four_pred
        four_pred_fine = four_pred_fine * resize_ratio + four_pred
        four_preds_list = four_preds_list + four_preds_list_fine
        return four_preds_list, four_pred_fine

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.image_1, self.image_2, self.fake_warped_image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        if self.args.GAN_mode == 'macegancross':
            pred_fake = nn.Sigmoid()(pred_fake)
        if self.args.GAN_mode in ['vanilla', 'lsgan']:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
        elif self.args.GAN_mode == 'macegan' or self.args.GAN_mode == 'macegancross':
            mace_ = (self.flow_4cor - self.four_pred)**2
            mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
            self.mace_vec_fake = torch.exp(self.args.ue_alpha * torch.mean(torch.mean(mace_, dim=1), dim=1)).detach() # exp(-0.1x)
            self.loss_D_fake = self.criterionGAN(pred_fake, self.mace_vec_fake)
        else:
            raise NotImplementedError()
        # Real
        real_AB = torch.cat((self.image_1, self.image_2, self.real_warped_image_2), 1)
        pred_real = self.netD(real_AB)
        if self.args.GAN_mode == 'macegancross':
            pred_fake = nn.Sigmoid()(pred_fake)
        if self.args.GAN_mode in ['vanilla', 'lsgan']:
            self.loss_D_real = self.criterionGAN(pred_real, True)
        elif self.args.GAN_mode == 'macegan' or self.args.GAN_mode == 'macegancross':
            self.mace_vec_real = torch.ones((real_AB.shape[0])).to(self.args.device)
            self.loss_D_real = self.criterionGAN(pred_real, self.mace_vec_real)
        else:
            raise NotImplementedError()
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.metrics["D_loss"] = self.loss_D.cpu().item()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.flow_gt, self.args.gamma, self.args, self.metrics) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Homo * self.G_loss_lambda
        self.metrics["G_loss"] = self.loss_G.cpu().item()
        if self.args.use_ue:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.image_1, self.image_2, self.fake_warped_image_2), 1)
            pred_fake = self.netD(fake_AB)
            if self.args.GAN_mode == 'macegancross':
                pred_fake = nn.Sigmoid()(pred_fake)
            if self.args.GAN_mode in ['vanilla', 'lsgan']:
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            elif self.args.GAN_mode == 'macegan' or self.args.GAN_mode == 'macegancross':
                self.loss_G_GAN = self.criterionGAN(pred_fake, self.mace_vec_fake) # Try not real
            else:
                raise NotImplementedError()
            self.loss_G = self.loss_G + self.loss_G_GAN
            self.metrics["GAN_loss"] = self.loss_G_GAN.cpu().item()
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward(use_raw_input = (self.args.train_ue_method == 'train_only_ue_raw_input'), noise_std=self.args.noise_std, sample_method=self.args.sample_method) # Calculate Fake A
        self.metrics = dict()
        # update D
        if self.args.use_ue:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            nn.utils.clip_grad_norm_(self.netD.parameters(), self.args.clip)
            self.optimizer_D.step()          # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        # update G
        if not self.args.train_ue_method in ['train_only_ue', 'train_only_ue_raw_input']:
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.args.clip)
            self.optimizer_G.step()             # update G's weights
        return self.metrics

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler_G.step()
        if self.args.use_ue:
            self.scheduler_D.step()

def mywarp(x, flow_pred, four_point_org_single):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    flow_4cor = torch.zeros((flow_pred.shape[0], 2, 2, 2)).to(flow_pred.device)
    flow_4cor[:, :, 0, 0] = flow_pred[:, :, 0, 0]
    flow_4cor[:, :, 0, 1] = flow_pred[:, :, 0, -1]
    flow_4cor[:, :, 1, 0] = flow_pred[:, :, -1, 0]
    flow_4cor[:, :, 1, 1] = flow_pred[:, :, -1, -1]

    four_point_1 = torch.zeros((flow_pred.shape[0], 2, 2, 2)).to(flow_pred.device)
    t_tensor = flow_4cor
    four_point_1[:, :, 0, 0] = t_tensor[:, :, 0, 0] + four_point_org_single[:, :, 0, 0]
    four_point_1[:, :, 0, 1] = t_tensor[:, :, 0, 1] + four_point_org_single[:, :, 0, 1]
    four_point_1[:, :, 1, 0] = t_tensor[:, :, 1, 0] + four_point_org_single[:, :, 1, 0]
    four_point_1[:, :, 1, 1] = t_tensor[:, :, 1, 1] + four_point_org_single[:, :, 1, 1]
    
    four_point_org = four_point_org_single.repeat(flow_pred.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
    four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous() 
    H = tgm.get_perspective_transform(four_point_org, four_point_1)
    warped_image = tgm.warp_perspective(x, H, (x.shape[2], x.shape[3]))
    return warped_image
    
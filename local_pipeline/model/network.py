import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
import kornia.geometry.bbox as bbox
from update import GMA
from extractor import BasicEncoderQuarter
from corr import CorrBlock
from utils import coords_grid, sequence_loss, single_loss, fetch_optimizer, warp
import os
import sys
from model.pix2pix_networks.networks import GANLoss, NLayerDiscriminator
from model.sync_batchnorm import convert_model
import wandb
import torchvision
import random
import time
import logging
from model.baseline import DHN, LocalTrans

autocast = torch.cuda.amp.autocast
class IHN(nn.Module):
    def __init__(self, args, first_stage):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128
        self.first_stage = first_stage
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        if self.args.lev0:
            sz = self.args.resize_width // 4
            self.update_block_4 = GMA(self.args, sz, first_stage)
        
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
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.args.resize_width//4-1, steps=self.args.resize_width//4), torch.linspace(0, self.args.resize_width//4-1, steps=self.args.resize_width//4))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.args.resize_width//4 * self.args.resize_width//4))),
                           dim=0).unsqueeze(0).repeat(H.shape[0], 1, 1).to(four_point.device)
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

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=3, corr_level=2, corr_radius=4):
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # time1 = time.time()
        with autocast(enabled=self.args.mixed_precision):
            # fmap1_64, fmap1_128 = self.fnet1(image1)
            # fmap2_64, _ = self.fnet1(image2)
            if not self.args.fnet_cat:
                fmap1_64 = self.fnet1(image1)
                fmap2_64 = self.fnet1(image2)
            else:
                fmap_64 = self.fnet1(torch.cat([image1, image2], dim=0))
                fmap1_64 = fmap_64[:image1.shape[0]]
                fmap2_64 = fmap_64[image1.shape[0]:]
        # time2 = time.time()
        # print("Time for fnet1: " + str(time2 - time1) + " seconds") # 0.004 + # 0.004

        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()

        # print(fmap1.shape, fmap2.shape)
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=corr_level, radius=corr_radius)
        coords0, coords1 = self.initialize_flow_4(image1)
        # print(coords0.shape, coords1.shape)
        sz = fmap1_64.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        four_point_predictions = []
        if self.first_stage and self.args.use_ue and self.args.D_net=="ue_branch":
            four_point_ue = []
        # time1 = time.time()
        for itr in range(iters_lev0):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            # print(corr.shape, flow.shape)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.weight:
                    delta_four_point, weight = self.update_block_4(corr, flow)
                else:
                    delta_four_point = self.update_block_4(corr, flow)
                    
            four_point_disp =  four_point_disp + delta_four_point[:, :2]
            if self.first_stage and self.args.use_ue and self.args.D_net=="ue_branch":
                four_point_ue.append(delta_four_point[:, 2])
            four_point_predictions.append(four_point_disp)
            coords1 = self.get_flow_now_4(four_point_disp)
        # time2 = time.time()
        # print("Time for iterative: " + str(time2 - time1) + " seconds") # 0.12

        if self.first_stage and self.args.use_ue and self.args.D_net=="ue_branch":
            return four_point_predictions, four_point_disp, four_point_ue
        else:
            return four_point_predictions, four_point_disp

arch_list = {"IHN": IHN,
             "DHN": DHN,
             "LocalTrans": LocalTrans,
             }

class STHN():
    def __init__(self, args, for_training=False):
        super().__init__()
        self.args = args
        self.device = args.device
        self.four_point_org_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_single[:, :, 0, 1] = torch.Tensor([self.args.resize_width - 1, 0]).to(self.device)
        self.four_point_org_single[:, :, 1, 0] = torch.Tensor([0, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_single[:, :, 1, 1] = torch.Tensor([self.args.resize_width - 1, self.args.resize_width - 1]).to(self.device)
        self.four_point_org_large_single = torch.zeros((1, 2, 2, 2)).to(self.device)
        self.four_point_org_large_single[:, :, 0, 0] = torch.Tensor([0, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 0, 1] = torch.Tensor([self.args.database_size - 1, 0]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 0] = torch.Tensor([0, self.args.database_size - 1]).to(self.device)
        self.four_point_org_large_single[:, :, 1, 1] = torch.Tensor([self.args.database_size - 1, self.args.database_size - 1]).to(self.device) # Only to calculate flow so no -1
        self.netG = arch_list[args.arch](args, True)
        if args.two_stages:
            corr_level = args.corr_level
            args.corr_level = 2
            self.netG_fine = IHN(args, False)
            args.corr_level = corr_level
            if args.restore_ckpt is not None and not args.finetune:
                self.set_requires_grad(self.netG, False)
        if args.use_ue:
            if args.D_net == 'patchGAN':
                self.netD = NLayerDiscriminator(6, norm="instance") # satellite=3 thermal=3 warped_thermal=3. norm should be instance?
            elif args.D_net == 'patchGAN_deep':
                self.netD = NLayerDiscriminator(6, n_layers=4, norm="instance")
            elif args.D_net == "ue_branch":
                pass
            else:
                raise NotImplementedError()
            self.criterionGAN = GANLoss(args.GAN_mode, bce_weight=args.bce_weight if args.GAN_mode=="vanilla_rej" else 1.0).to(args.device)
        self.criterionAUX = sequence_loss if self.args.arch == "IHN" else single_loss
        if for_training:
            if args.two_stages:
                if args.restore_ckpt is None or args.finetune:
                    self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()) + list(self.netG_fine.parameters()))
                else:
                    self.optimizer_G, self.scheduler_G = fetch_optimizer(args,list(self.netG_fine.parameters()))
            else:
                self.optimizer_G, self.scheduler_G = fetch_optimizer(args, list(self.netG.parameters()))
            if args.use_ue and args.D_net != "ue_branch":
                self.optimizer_D, self.scheduler_D = fetch_optimizer(args, list(self.netD.parameters()))
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
    
    def set_input(self, A, B, flow_gt=None, A_ori=None):
        self.image_1 = A.to(self.device, non_blocking=True)
        self.image_2 = B.to(self.device, non_blocking=True)
        self.image_1_ori = A_ori.to(self.device, non_blocking=True)
        self.flow_gt = flow_gt.to(self.device, non_blocking=True)
        if self.flow_gt is not None:
            self.real_warped_image_2 = mywarp(self.image_2, self.flow_gt, self.four_point_org_single)
            self.flow_4cor = torch.zeros((self.flow_gt.shape[0], 2, 2, 2)).to(self.flow_gt.device)
            self.flow_4cor[:, :, 0, 0] = self.flow_gt[:, :, 0, 0]
            self.flow_4cor[:, :, 0, 1] = self.flow_gt[:, :, 0, -1]
            self.flow_4cor[:, :, 1, 0] = self.flow_gt[:, :, -1, 0]
            self.flow_4cor[:, :, 1, 1] = self.flow_gt[:, :, -1, -1]
        else:
            self.real_warped_image_2 = None

    def predict_uncertainty(self, GAN_mode='vanilla', for_training=False):
        if self.args.D_net == "ue_branch":
            fake_AB_conf = self.four_ue[-1]
        else:
            if self.args.two_stages:
                fake_AB = torch.cat((self.image_1_crop, self.image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            else:
                fake_AB = torch.cat((self.image_1, self.image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB_conf = self.netD(fake_AB)
            if GAN_mode in ['vanilla', 'vanilla_rej'] and not for_training:
                fake_AB_conf = nn.Sigmoid()(fake_AB_conf)
            elif for_training:
                pass
            else:
                raise NotImplementedError()
        return fake_AB_conf
        
    def forward(self, use_raw_input=False, noise_std=0, sample_method="target_raw"):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not use_raw_input:
            # time1 = time.time()
            if self.args.use_ue and self.args.D_net == "ue_branch":
                self.four_preds_list, self.four_pred, self.four_ue = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
            else:
                self.four_preds_list, self.four_pred = self.netG(image1=self.image_1, image2=self.image_2, iters_lev0=self.args.iters_lev0, corr_level=self.args.corr_level)
            # time2 = time.time()
            # logging.debug("Time for 1st forward pass: " + str(time2 - time1) + " seconds")
            if self.args.two_stages:
                # self.four_pred = self.flow_4cor # DEBUG
                # self.four_preds_list[-1] = self.four_pred # DEBUG
                # self.four_preds_list[-1] = torch.zeros_like(self.four_pred).to(self.four_pred.device) # DEBUG
                # time1 = time.time()
                self.image_1_crop, delta, self.flow_bbox = self.get_cropped_st_images(self.image_1_ori, self.four_pred, self.args.fine_padding, self.args.detach, self.args.augment_two_stages)
                # time2 = time.time()
                # logging.debug("Time for crop: " + str(time2 - time1) + " seconds")
                # time1 = time.time()
                self.four_preds_list_fine, self.four_pred_fine = self.netG_fine(image1=self.image_1_crop, image2=self.image_2, iters_lev0=self.args.iters_lev1)
                # time2 = time.time()
                # logging.debug("Time for 2nd forward pass: " + str(time2 - time1) + " seconds")
                # self.four_pred_fine = torch.zeros_like(self.four_pred).to(self.four_pred.device) # DEBUG
                # self.four_preds_list_fine[-1] = self.four_pred_fine # DEBUG
                self.four_preds_list, self.four_pred = self.combine_coarse_fine(self.four_preds_list, self.four_pred, self.four_preds_list_fine, self.four_pred_fine, delta, self.flow_bbox)
            self.fake_warped_image_2 = mywarp(self.image_2, self.four_pred, self.four_point_org_single) # Comment for performance evaluation
        elif self.args.GAN_mode == "vanilla_rej":
            pass
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

    def get_cropped_st_images(self, image_1_ori, four_pred, fine_padding, detach=True, augment_two_stages=0):
        # From four_pred to bbox coordinates
        four_point = four_pred + self.four_point_org_single
        x = four_point[:, 0]
        y = four_point[:, 1]
        # Make it same scale as image_1_ori
        alpha = self.args.database_size / self.args.resize_width
        x[:, :, 0] = x[:, :, 0] * alpha
        x[:, :, 1] = (x[:, :, 1] + 1) * alpha
        y[:, 0, :] = y[:, 0, :] * alpha
        y[:, 1, :] = (y[:, 1, :] + 1) * alpha
        # Crop
        left = torch.min(x.view(x.shape[0], -1), dim=1)[0]  # B
        right = torch.max(x.view(x.shape[0], -1), dim=1)[0] # B
        top = torch.min(y.view(y.shape[0], -1), dim=1)[0]   # B
        bottom = torch.max(y.view(y.shape[0], -1), dim=1)[0] # B
        if augment_two_stages!=0:
            if self.args.augment_type == "bbox":
                left += (torch.rand(left.shape).to(left.device) * 2 - 1) * augment_two_stages
                right += (torch.rand(right.shape).to(right.device) * 2 - 1) * augment_two_stages
                top += (torch.rand(top.shape).to(top.device) * 2 - 1) * augment_two_stages
                bottom += (torch.rand(bottom.shape).to(bottom.device) * 2 - 1) * augment_two_stages
            w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0] # B
            c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1) # B, 2
            if self.args.augment_type == "center":
                w += torch.rand(w.shape).to(w.device) * augment_two_stages # only expand?
                c += (torch.rand(c.shape).to(c.device) * 2 - 1) * augment_two_stages
        else:
            w = torch.max(torch.stack([right-left, bottom-top], dim=1), dim=1)[0] # B
            c = torch.stack([(left + right)/2, (bottom + top)/2], dim=1) # B, 2
        w_padded = w + 2 * fine_padding # same as ori scale
        crop_top_left = c + torch.stack([-w_padded / 2, -w_padded / 2], dim=1) # B, 2 = x, y
        x_start = crop_top_left[:, 0] # B
        y_start = crop_top_left[:, 1] # B
        # Do not use bbox_generator because it will repeat to reduce 1 for end index
        bbox_s = bbox.bbox_generator(x_start, y_start, w_padded, w_padded)
        delta = (w_padded / self.args.resize_width).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        image_1_crop = tgm.crop_and_resize(image_1_ori, bbox_s, (self.args.resize_width, self.args.resize_width)) # It will be padded when it is out of boundary
        # swap bbox_s
        bbox_s_swap = torch.stack([bbox_s[:, 0], bbox_s[:, 1], bbox_s[:, 3], bbox_s[:, 2]], dim=1)
        four_cor_bbox = bbox_s_swap.permute(0, 2, 1). view(-1, 2, 2, 2)
        flow_bbox = four_cor_bbox - self.four_point_org_large_single
        if detach:
            image_1_crop = image_1_crop.detach()
            delta = delta.detach()
            flow_bbox = flow_bbox.detach()
        return image_1_crop, delta, flow_bbox
    
    def combine_coarse_fine(self, four_preds_list, four_pred, four_preds_list_fine, four_pred_fine, delta, flow_bbox):
        alpha = self.args.database_size / self.args.resize_width
        kappa = delta / alpha
        four_preds_list_fine = [four_preds_list_fine_single * kappa + flow_bbox / alpha for four_preds_list_fine_single in four_preds_list_fine]
        four_pred_fine = four_pred_fine * kappa + flow_bbox / alpha
        four_preds_list = four_preds_list + four_preds_list_fine
        return four_preds_list, four_pred_fine

    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     if self.args.two_stages:
    #         fake_AB = torch.cat((self.image_1_crop, self.image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     else:
    #         fake_AB = torch.cat((self.image_1, self.image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.netD(fake_AB.detach())
    #     if self.args.GAN_mode in ['vanilla', 'lsgan']:
    #         self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     elif self.args.GAN_mode == 'macegan':
    #         mace_ = (self.flow_4cor - self.four_pred)**2
    #         mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
    #         self.mace_vec_fake = torch.exp(self.args.ue_alpha * torch.mean(torch.mean(mace_, dim=1), dim=1)).detach() # exp(-0.1x)
    #         self.loss_D_fake = self.criterionGAN(pred_fake, self.mace_vec_fake)
    #     else:
    #         raise NotImplementedError()
    #     # Real
    #     real_AB = torch.cat((self.image_1, self.image_2, self.real_warped_image_2), 1)
    #     pred_real = self.netD(real_AB)
    #     if self.args.GAN_mode in ['vanilla', 'lsgan']:
    #         self.loss_D_real = self.criterionGAN(pred_real, True)
    #     elif self.args.GAN_mode == 'macegan':
    #         self.mace_vec_real = torch.ones((real_AB.shape[0])).to(self.args.device)
    #         self.loss_D_real = self.criterionGAN(pred_real, self.mace_vec_real)
    #     else:
    #         raise NotImplementedError()
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()
    #     self.metrics["D_loss"] = self.loss_D.cpu().item()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.args.two_stages:
            fake_AB = torch.cat((self.image_1_crop, self.image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        else:
            fake_AB = torch.cat((self.image_1, self.image_2), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        if self.args.GAN_mode in ['vanilla', 'lsgan']:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
        elif self.args.GAN_mode == 'macegan' and self.args.D_net != "ue_branch":
            mace_ = (self.flow_4cor - self.four_pred)**2
            mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
            self.mace_vec_fake = torch.exp(self.args.ue_alpha * torch.mean(torch.mean(mace_, dim=1), dim=1)).detach() # exp(-0.1x)
            self.loss_D_fake = self.criterionGAN(pred_fake, self.mace_vec_fake)
        elif self.args.GAN_mode == 'vanilla_rej':
            flow_ = (self.flow_4cor)**2
            flow_ = ((flow_[:,0,:,:] + flow_[:,1,:,:])**0.5)
            flow_vec = torch.mean(torch.mean(flow_, dim=1), dim=1)
            flow_bool = torch.ones_like(flow_vec)
            alpha = self.args.database_size / self.args.resize_width
            flow_bool[flow_vec >= (self.args.rej_threshold / alpha)] = 0.0
            self.loss_D_fake = self.criterionGAN(pred_fake, flow_bool)
        else:
            raise NotImplementedError()
        self.loss_D = self.loss_D_fake
        self.loss_D.backward()
        self.metrics["D_loss"] = self.loss_D.cpu().item()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        if self.args.use_ue and self.args.D_net == "ue_branch":
            self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.flow_gt, self.args.gamma, self.args, self.metrics, four_ue=self.four_ue) 
        else:
            self.loss_G_Homo, self.metrics = self.criterionAUX(self.four_preds_list, self.flow_gt, self.args.gamma, self.args, self.metrics) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Homo * self.G_loss_lambda
        self.metrics["G_loss"] = self.loss_G.cpu().item()
        if self.args.use_ue:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.image_1, self.image_2), 1)
            if self.args.D_net != "ue_branch":
                pred_fake = self.netD(fake_AB)
            if self.args.GAN_mode in ['vanilla', 'lsgan']:
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            elif self.args.GAN_mode == 'macegan' and self.args.D_net != "ue_branch":
                self.loss_G_GAN = self.criterionGAN(pred_fake, self.mace_vec_fake) # Try not real
            elif self.args.GAN_mode == 'vanilla_rej' or self.args.D_net == "ue_branch":
                self.loss_G_GAN = 0
            else:
                raise NotImplementedError()
            self.loss_G = self.loss_G + self.loss_G_GAN
            try:
                self.metrics["GAN_loss"] = self.loss_G_GAN.cpu().item()
            except AttributeError:
                self.metrics["GAN_loss"] = 0
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
        if self.args.use_ue and self.args.D_net != "ue_branch":
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
            if self.args.restore_ckpt is None or self.args.finetune:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.args.clip)
            if self.args.two_stages:
                nn.utils.clip_grad_norm_(self.netG_fine.parameters(), self.args.clip)
            self.optimizer_G.step()             # update G's weights
        return self.metrics

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.scheduler_G.step()
        if self.args.use_ue and self.args.D_net != "ue_branch":
            self.scheduler_D.step()

def mywarp(x, flow_pred, four_point_org_single):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if flow_pred.shape[-1] != 2:
        flow_4cor = torch.zeros((flow_pred.shape[0], 2, 2, 2)).to(flow_pred.device)
        flow_4cor[:, :, 0, 0] = flow_pred[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_pred[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_pred[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_pred[:, :, -1, -1]
    else:
        flow_4cor = flow_pred

    four_point_1 = flow_4cor + four_point_org_single
    
    four_point_org = four_point_org_single.repeat(flow_pred.shape[0],1,1,1).flatten(2).permute(0, 2, 1).contiguous() 
    four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous() 
    try:
        H = tgm.get_perspective_transform(four_point_org, four_point_1)
    except Exception:
        logging.debug("No solution")
        H = torch.eye(3).to(four_point_org.device).repeat(four_point_1.shape[0],1,1)
    warped_image = tgm.warp_perspective(x, H, (x.shape[2], x.shape[3]))
    return warped_image
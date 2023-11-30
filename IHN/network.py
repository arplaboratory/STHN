import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm
from update import GMA
from extractor import BasicEncoderQuarter
from corr import CorrBlock
from utils import *
from model.pix2pix_networks.networks import GANLoss, get_scheduler, NLayerDiscriminator

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

    def forward(self, image1, image2 , iters_lev0 = 6, iters_lev1=3, test_mode=False):
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

        if test_mode:
            return four_point_disp
        return four_point_predictions


class STHEGAN():
    def __init__(self, args, homo_model, input_channel_num, output_channel_num):
        super().__init__()
        self.device = args.device
        self.G_net = homo_model
        if args.D_net == 'patchGAN':
            self.netD = NLayerDiscriminator(input_channel_num + output_channel_num)
        elif args.D_net == 'patchGAN_deep':
            self.netD = NLayerDiscriminator(input_channel_num + output_channel_num, n_layers=4)
        else:
            raise NotImplementedError()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.scheduler_G = get_scheduler(self.optimizer_G, args)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.scheduler_D = get_scheduler(self.optimizer_D, args)
        self.criterionGAN = GANLoss(args.GAN_mode).to(args.device)
        self.criterionAUX = None

    def setup(self):
        if hasattr(self, 'netD'):
            self.netD = self.init_net(self.netD)
        self.netG = self.init_net(self.netG)

    def init_net(self, model):
        model = torch.nn.DataParallel(model)
        if torch.cuda.device_count() >= 2:
            # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
            model = convert_model(model)
            model = model.to(self.device)
        return model
    
    def set_input(self, A, B, flow_gt):
        self.image_1 = A.to(self.device)
        self.image_2 = B.to(self.device)
        flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        self.flow_4cor = flow_4cor

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        four_pred = self.netG(self.image1, self.image2, iters_lev0=self.args.iters_lev0, iters_lev1=self.args.iters_lev1)
        self.real_B = None
        # self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionAUX(self.fake_B, self.real_B) * self.G_loss_lambda
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
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
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizer_G.param_groups[0]['lr']
        self.scheduler_G.step()
        self.scheduler_D.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        logging.debug('learning rate %.7f -> %.7f' % (old_lr, lr))



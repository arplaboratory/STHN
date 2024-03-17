
import os
import torch
import logging
import torchvision
from torch import nn
import torchvision
from os.path import join
from transformers import ViTModel
from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock
from model.functional import ReverseLayerF
from model.pix2pix_networks.networks import UnetGenerator, GANLoss, NLayerDiscriminator, get_scheduler
from model.sync_batchnorm import convert_model

from model.Deit import deit_base_distilled_patch16_384, deit_small_distilled_patch16_224

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        self.self_att = False
        self.DA = args.DA

        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            if args.aggregation in ["netvlad", "crn"]:
                args.features_dim = 65536
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
        
        if args.aggregation in ["netvlad", "crn"] and hasattr(args, "conv_output_dim") and args.conv_output_dim != None:
            # Concatenate conv layer to the aggregation layer
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)
            logging.debug(f"Last conv layer dim: {actual_conv_output_dim}")
            self.conv_layer = nn.Conv2d(args.features_dim, actual_conv_output_dim, 1)
            args.features_dim = actual_conv_output_dim

        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                           channel_inner=args.channel_bottleneck)]* args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True

    def create_domain_classifier(self, args):
        domain_classifier = None
        domain_classifier = nn.Sequential(nn.Linear(args.fc_output_dim, 100, bias=False),
                                                nn.BatchNorm1d(100),
                                                nn.ReLU(True),
                                                nn.Linear(100, 2),
                                                nn.LogSoftmax(dim=1))
        return domain_classifier


    def forward(self, x, is_train=False, alpha=None):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        x_after = self.aggregation(x)
        if is_train is True:
            if not self.DA:
                return x_after
            else:
                reverse_x = ReverseLayerF.apply(x_after, alpha)
                return x_after, reverse_x
        return x_after


def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        if hasattr(args, "conv_output_dim") and args.conv_output_dim is not None:
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)
            return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=actual_conv_output_dim,
                                    work_with_tokens=args.work_with_tokens)
        else:
            return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                    work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        if hasattr(args, "conv_output_dim") and args.conv_output_dim is not None:
            actual_conv_output_dim = int(args.conv_output_dim / args.netvlad_clusters)
            return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=actual_conv_output_dim)
        else:
            return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation == 'none'\
            or args.aggregation == 'cls' \
            or args.aggregation == 'seqpool':
        return nn.Identity()


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    else: raise NotImplementedError()
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        update_state_dict = dict()
        for key, value in state_dict.items():
            remove_prefix_key = key.replace('module.encoder.', '')
            update_state_dict[remove_prefix_key] = value
        update_state_dict.pop('fc.weight', None)
        update_state_dict.pop('fc.bias', None)
        state_dict = update_state_dict
    model.load_state_dict(state_dict, strict=False)
    return model


def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        if not args.unfreeze:
            for name, child in backbone.named_children():
                # Freeze layers before conv_3
                if name == "layer3":
                    break
                for params in child.parameters():
                    params.requires_grad = False
        if args.backbone.endswith("conv4"):
            if not args.unfreeze:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x of the {args.backbone.split('conv')[0]} (remove conv5_x)")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            if not args.unfreeze:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}, freeze the previous ones")
            else:
                logging.debug(f"Train only conv4_x and conv5_x of the {args.backbone.split('conv')[0]}")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the vgg16, freeze the previous ones")
        else:
            logging.debug("Train last layers of the vgg16")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        if not args.unfreeze:
            logging.debug("Train last layers of the alexnet, freeze the previous ones")
        else:
            logging.debug("Train last layers of the alexnet")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        if args.resize[0] == 224:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif args.resize[0] == 384:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        else:
            raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]


class pix2pix():
    def __init__(self, args, input_channel_num, output_channel_num, for_training=False):
        super().__init__()
        if args.G_net == 'unet':
            self.netG = UnetGenerator(input_channel_num, output_channel_num, 8, norm=args.GAN_norm, upsample=args.GAN_upsample, use_tanh=args.G_tanh)
        elif args.G_net == 'unet_deep':
            self.netG = UnetGenerator(input_channel_num, output_channel_num, 9, norm=args.GAN_norm, upsample=args.GAN_upsample, use_tanh=args.G_tanh)
        else:
            raise NotImplementedError()
        self.device = args.device
        if for_training:
            if args.D_net == 'patchGAN':
                self.netD = NLayerDiscriminator(input_channel_num + output_channel_num)
            elif args.D_net == 'patchGAN_deep':
                self.netD = NLayerDiscriminator(input_channel_num + output_channel_num, n_layers=4)
            else:
                raise NotImplementedError()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
            self.scheduler_G = get_scheduler(self.optimizer_G, args)
            self.scheduler_D = get_scheduler(self.optimizer_D, args)
            self.G_loss_lambda = args.G_loss_lambda
            self.criterionGAN = GANLoss(args.GAN_mode).to(args.device)
            self.criterionAUX = torch.nn.L1Loss()

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
    
    def set_input(self, A, B):
        self.real_A = A.to(self.device)
        self.real_B = B.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

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


class GeoLocalizationNetRerank(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.arch_name = args.backbone
        self.out_dim = args.local_dim
        if args.backbone.startswith("deit"):
            if args.backbone == 'deitBase':
                self.backbone = deit_base_distilled_patch16_384(img_size=args.resize, num_classes=args.fc_output_dim, embed_layer=AnySizePatchEmbed)
            else:
                self.backbone = deit_small_distilled_patch16_224(img_size=args.resize, num_classes=args.fc_output_dim, embed_layer=AnySizePatchEmbed)
            args.features_dim = args.fc_output_dim
            if args.hypercolumn:
                self.hyper_s = args.hypercolumn // 100
                self.hyper_e = args.hypercolumn % 100
                self.local_head = nn.Linear(self.backbone.embed_dim*(self.hyper_e-self.hyper_s), self.out_dim, bias=True)
            else:
                self.local_head = nn.Linear(self.backbone.embed_dim, self.out_dim, bias=True)
        else:
            self.backbone = get_backbone(args)
            self.aggregation = get_aggregation(args)
            self.self_att = False

            if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
                if args.l2 == "before_pool":
                    self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
                elif args.l2 == "after_pool":
                    self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
                elif args.l2 == "none":
                    self.aggregation = nn.Sequential(self.aggregation, Flatten())

            if args.fc_output_dim != None:
                # Concatenate fully connected layer to the aggregation layer
                self.aggregation = nn.Sequential(self.aggregation,
                                                 nn.Linear(args.features_dim, args.fc_output_dim),
                                                 L2Norm())
                args.features_dim = args.fc_output_dim
            if args.non_local:
                non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                                channel_inner=args.channel_bottleneck)] * args.num_non_local
                self.non_local = nn.Sequential(*non_local_list)
                self.self_att = True
            if args.hypercolumn:
                self.local_head = nn.Linear(1856, self.out_dim, bias=True)
            else:
                self.local_head = nn.Linear(1024, self.out_dim, bias=True)
        # ==================================================================
        self.local_head.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head.bias.data.zero_()
        self.multi_out = args.num_local
        self.single = False
        if args.rerank_model == 'r2former':
            self.Reranker = R2Former(decoder_depth=6, decoder_num_heads=4,
                                     decoder_embed_dim=32, decoder_mlp_ratio=4,
                                     decoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                     num_classes=2, num_patches=2 * self.multi_out,
                                     input_dim=args.fc_output_dim, num_corr=5)
        else:
            print('rerank_model not implemented!')
            raise Exception

    def forward_ori(self, x):
        x = self.backbone(x)
        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            x = x.last_hidden_state[:, 0, :]
            return x
        x = self.aggregation(x)
        return x

    def res_forward(self, x):
        # print(self.backbone)
        # raise Exception
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        x0 = x*1 #.detach()

        x = self.backbone[4](x)
        x1 = x*1 #.detach()
        x = self.backbone[5](x)
        x2 = x*1 #.detach()
        x = self.backbone[6](x)
        x3 = x*1
        x = self.backbone[7](x)
        x4 = x*1  #.detach()

        if self.args.hypercolumn:
            B,C, H, W = x3.shape
            local_feature = torch.cat([
                F.interpolate(x0, size=(H, W), mode='bicubic'), # 64
                F.interpolate(x1, size=(H, W), mode='bicubic'), # 256
                F.interpolate(x2, size=(H, W), mode='bicubic'), # 512
                x3, # 1024
                # F.interpolate(x4, size=(H, W), mode='bicubic'),
            ],dim=1)
        else:
            local_feature = x3

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x, local_feature, x4

    def forward_cnn(self, x):
        # with torch.no_grad():
        B,_, H, W = x.shape
        query_img = x.clone()
        x, feature, feature_last = self.res_forward(x)
        x = self.aggregation(x)

        _, C, f_H, f_W = feature.shape
        assert f_H == np.ceil(H/16).astype(int) and f_W == np.ceil(W/16).astype(int)
        feature_reshape = feature.permute((0,2,3,1)).reshape(B, f_H*f_W, C)
        # print(feature_last.shape, feature_reshape.shape, query_img.shape, H//32, W//32)
        feature_last_reshape = feature_last.permute((0,2,3,1)).reshape(B, np.ceil(H/32).astype(int)*np.ceil(W/32).astype(int), 2048)
        feature_last_reshape = F.normalize(feature_last_reshape, p=2, dim=2)
        # print(self.aggregation)
        fc_weight = self.aggregation[1].weight.t()
        # fc_weight = torch.eye(2048, dtype=torch.float32).cuda()
        sim = torch.matmul(feature_last_reshape.clamp(min=1e-6), fc_weight)
        last_map = (sim.clamp(min=1e-6)).sum(dim=2)  # /sim.max(dim=1,keepdim=True)[0]
        last_map_reshape = F.interpolate(last_map.reshape([B, 1, np.ceil(H/32).astype(int), np.ceil(W/32).astype(int)]),
                                        size=(np.ceil(H/16).astype(int), np.ceil(W/16).astype(int)), mode='bicubic')
        last_map = last_map_reshape.reshape(B, np.ceil(H/16).astype(int)*np.ceil(W/16).astype(int))
        # print(query_img.shape, x.shape, feature.shape, feature_reshape.shape)
        # print(sim.shape, last_map.shape, last_map_reshape.shape)

        order = torch.argsort(last_map, dim=1)
        multi_out = np.minimum(order.shape[1], self.multi_out)
        if order.shape[1] < self.multi_out:
            print(order.shape, last_map.shape, last_map)
        local_features = torch.gather(input=feature_reshape,
                                      index=order[:, -multi_out:].unsqueeze(2).repeat(1, 1, feature_reshape.shape[2]),
                                      dim=1)

        HW = max(H, W)
        # HW = 512.
        x_xy = torch.cat([(order[:, -multi_out:].unsqueeze(2) % np.ceil(W/16).astype(int) * 16 + 8) / 1. / HW,
                          (order[:, -multi_out:].unsqueeze(2) // np.ceil(W/16).astype(int) * 16 + 8) / 1. / HW], dim=2)
        x_attention = torch.sort(last_map, dim=1)[0][:, -multi_out:]
        x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                                 x_xy.shape[1], 1)
        if self.args.finetune:
            local_features = self.local_head(local_features.reshape(B*multi_out, C)).reshape(B, multi_out, self.out_dim)
        else:
            local_features = self.local_head(local_features.detach().reshape(B * multi_out, C)).reshape(B, multi_out,
                                                                                               self.out_dim)
        if self.single:
            return x
        else:
            return x, torch.flip(torch.cat([x_xy, x_attention, local_features], dim=2),dims=(1,))

    def forward_deit(self, x):
        # with torch.no_grad():
        B, _, H, W = x.shape
        x_ori = x.detach()
        x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.backbone.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if H != self.backbone.patch_embed.img_size[0] or W != self.backbone.patch_embed.img_size[1]:
            grid_size = [self.backbone.patch_embed.img_size[0]//16, self.backbone.patch_embed.img_size[1]//16]
            matrix = self.backbone.pos_embed[:, 2:].reshape((1, grid_size[0], grid_size[1],self.backbone.embed_dim)).permute((0, 3, 1, 2))
            new_size = max(H//16, W//16)
            if grid_size[0] >= new_size and grid_size[1] >= new_size:
                re_matrix = matrix[:, :, (grid_size[0]//2 - new_size//2):(grid_size[0]//2 - new_size//2 + new_size),
                            (grid_size[1]//2 - new_size//2):(grid_size[1]//2 - new_size//2+new_size)]
            else:
                re_matrix = pos_resize(matrix, (new_size, new_size))
            if H >= W:
                new_matrix = re_matrix[:, :, :, (new_size//2 - W//16//2):(new_size//2 - W//16//2 + W//16)].permute(0, 2, 3, 1).reshape([1, -1, self.backbone.pos_embed.shape[-1]])
            else:
                new_matrix = re_matrix[:, :, (new_size//2 - H//16//2):(new_size//2 - H//16//2 + H//16), :].permute(0, 2, 3, 1).reshape([1, -1, self.backbone.pos_embed.shape[-1]])
            # print(new_matrix.shape,H//16, W//16,new_size)
            new_pos_embed = torch.cat([self.backbone.pos_embed[:, :2], new_matrix], dim=1)
            x = x + new_pos_embed
        else:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        output_list = []

        for i, blk in enumerate(self.backbone.blocks):
            if (not self.single) and i == (len(self.backbone.blocks)-1):  # len(self.blocks)-1:
                output = x * 1
                y = blk.norm1(x)
                B, N, C = y.shape
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                last_map = (att[:, :, :2, 2:].detach()).sum(dim=1).sum(dim=1)
            x = blk(x)

            # to support hypercolumn, not used, can be removed.
            if (not self.single) and self.args.hypercolumn:
                if self.hyper_s <= i < self.hyper_e:
                    output_list.append(x*1.) # .detach()

        x = self.backbone.norm(x)

        x_cls = self.backbone.head(x[:, 0])
        x_dist = self.backbone.head_dist(x[:, 1])

        if self.single:
            return self.backbone.l2_norm((x_cls + x_dist) / 2)
        else:
            if self.args.hypercolumn:
                output = torch.cat(output_list, dim=2)
            order = torch.argsort(last_map, dim=1, descending=True)
            multi_out = np.minimum(order.shape[1], self.multi_out)
            local_features = torch.gather(input=output,
                                          index=order[:, :multi_out].unsqueeze(2).repeat(1, 1, output.shape[2]),
                                          dim=1)
            # compute attention and coordinates
            HW = max(H, W)
            x_xy = torch.cat([(order[:, :multi_out].unsqueeze(2) % np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW,
                              (order[:, :multi_out].unsqueeze(2) // np.ceil(W / 16).astype(int) * 16 + 8) / 1. / HW],
                             dim=2)
            x_attention = torch.sort(last_map, dim=1, descending=True)[0][:, :multi_out]
            x_attention = (x_attention / torch.max(x_attention, dim=1, keepdim=True)[0]).reshape(x_xy.shape[0],
                                                                                                 x_xy.shape[1], 1)
            if self.args.finetune:
                local_features = self.local_head(local_features.reshape(B * multi_out, -1)). \
                    reshape(B, multi_out, self.out_dim)
            else:
                local_features = self.local_head(local_features.detach().reshape(B * multi_out, -1)).\
                reshape(B, multi_out, self.out_dim)
            return self.backbone.l2_norm((x_cls + x_dist) / 2), torch.cat([x_xy, x_attention, local_features], dim=2)

    def forward(self, x):
        if self.args.backbone.startswith("deit"):
            return self.forward_deit(x)
        else:
            return self.forward_cnn(x)
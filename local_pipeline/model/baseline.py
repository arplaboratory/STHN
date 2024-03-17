import torch.nn as nn
import torch
from model.ATT.attention_layer import Correlation, AttentionLayer
import torch.nn.functional as F
class DHN(nn.Module):
    def __init__(self, args, first_stage):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.first_stage = first_stage
        self.layer1 = nn.Sequential(nn.Conv2d(6,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())        
        self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.fc1 = nn.Linear(128*32*32,1024)
        self.fc2 = nn.Linear(1024,8)

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=3, corr_level=2, corr_radius=4):
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        x = torch.cat((image1, image2), 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(-1,128* 32* 32)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.view(-1, 2, 2, 2)
        return [out], out
    
class conv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU())
    
    def forward(self, x):
        return self.conv(x)

class LocalTrans(nn.Module):
    def __init__(self, args, first_stage):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.first_stage = first_stage
        self.conv1 = nn.Sequential(conv3x3(3, 32), conv3x3(32, 32), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(conv3x3(32, 64), conv3x3(64, 64), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(conv3x3(64, 64), conv3x3(64, 64), nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(conv3x3(64, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2))

        self.transformer1 = AttentionLayer(128, 4, 32, 32, 5, 2)
        self.transformer2 = AttentionLayer(64, 2, 32, 32, 7, 3)
        self.transformer3 = AttentionLayer(64, 2, 32, 32, 9, 4)
        self.transformer4 = AttentionLayer(64, 2, 32, 32, 9, 4)
        self.transformer5 = AttentionLayer(64, 2, 32, 32, 9, 4)
        self.transformer = [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5]

        self.homo1 = nn.Sequential(conv3x3(25, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo2 = nn.Sequential(conv3x3(25, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo3 = nn.Sequential(conv3x3(81, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2),
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo4 = nn.Sequential(conv3x3(81, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2),
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))
        self.homo5 = nn.Sequential(conv3x3(81, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 128), conv3x3(128, 128), nn.MaxPool2d(2, 2),
            conv3x3(128, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2), 
            conv3x3(256, 256), conv3x3(256, 256), nn.MaxPool2d(2, 2),
            conv3x3(256, 256), conv3x3(256, 256), nn.AvgPool2d(2, 2), nn.Conv2d(256, 8, 1))

        self.homo_estim = [self.homo1, self.homo2, self.homo3, self.homo4, self.homo5]

        self.kernel_list = [5, 7, 9, 9, 9]
        self.pad_list = [2, 3, 4, 4, 4]
        self.scale_list = [16, 8, 4, 4, 4]
        self.bias_list = [2, 1, 0.5, 0.25, 0.125]

    def forward(self, image1, image2, iters_lev0 = 6, iters_lev1=3, corr_level=2, corr_radius=4):
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        x = image1
        y = image2
        device = x.device
        B, C, H, W = x.shape
        out_list = []
        
        for L in range(1):
            x, y = self.conv2(self.conv1(x)), self.conv2(self.conv1(y))
            if L <= 1:
                x, y = self.conv3(x), self.conv3(y)
            if L <= 0:
                x, y = self.conv4(x), self.conv4(y)

            transformer = self.transformer[L]
            x, y = transformer(x, y)
                
            scale = self.scale_list[L]
            corr = Correlation.apply(x.contiguous(), y.contiguous(), self.kernel_list[L], self.pad_list[L])
            corr = corr.permute(0, 3, 1, 2) / x.shape[1]
            homo_flow = self.homo_estim[L+1](corr) * scale * self.bias_list[L]
            out = homo_flow.reshape(B, 2, 2, 2)
            out_list.append(out)

        return [out], out
           

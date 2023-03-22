import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from .vgg import VGG

class LSPMLayer(Module):
    def __init__(self, in_planes, S):
        super(LSPMLayer, self).__init__()

        self.in_planes = in_planes
        self.S = S

        self.GAP      = nn.AdaptiveAvgPool2d(self.S)
        self.GAP_1X1  = nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False)
        self.GAP_Rule = nn.ReLU(inplace=True)
        self.conv     = nn.Conv2d(in_channels=in_planes, out_channels=S*S, kernel_size=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.size() #B=1 C=512 H=16 W=16

        x_reshape_T = x.view(B, -1, H*W).permute(0, 2, 1)           # B=1,HW=256,C=512
        x_reshape   = x.view(B, -1, H*W)                            # B=1,C=512,HW=256
        MM1         = torch.bmm(x_reshape_T, x_reshape)             # B=1, HW=256, HW=256     torch.bmm计算两个tensor的矩阵乘法
        MM1         = F.softmax(MM1, dim=-1)                        # B=1, HW=256, HW=256

        x_conv = self.conv(x).view(B, -1, H*W)                      # B=1, SS=1, HW=256
        MM2    = torch.bmm(x_conv, MM1)                             # B=1, SS=1, HW=256

        GAP = self.GAP(x)                                           # B=1 C=512 H=1 W=1
        GAP = self.GAP_1X1(GAP)                                     # B=1 C=512 H=1 W=1
        GAP = self.GAP_Rule(GAP)                                    # B=1 C=512 H=1 W=1
        GAP = GAP.view(B, -1, self.S*self.S)                        # B=1, C=512, SS=1

        MM3     = torch.bmm(GAP, MM2).view(B, C, H, W)              # B=1, C=512, H=16, W=16
        results = torch.add(MM3, x)                                 # B=1, C=512, H=16, W=16

        return results

class LSPM(Module):
    def __init__(self, in_planes):
        super(LSPM, self).__init__()
        self.in_planes = in_planes
        self.LSPM_1 = LSPMLayer(self.in_planes, 1)
        self.LSPM_2 = LSPMLayer(self.in_planes, 2)
        self.LSPM_3 = LSPMLayer(self.in_planes, 3)
        self.LSPM_6 = LSPMLayer(self.in_planes, 6)

        self.conv  = nn.Conv2d(5*self.in_planes, self.in_planes, 1, 1, bias=False)

    def forward(self, x):
        '''
        inputs:
        x:  input feature maps(B, C, H, W)
        returns:
        out:  B, C, H, W
        '''

        LSPM_1 = self.LSPM_1(x)      # B=1, C=512, H=16, W=16
        LSPM_2 = self.LSPM_2(x)      # B=1, C=512, H=16, W=16
        LSPM_3 = self.LSPM_3(x)      # B=1, C=512, H=16, W=16
        LSPM_6 = self.LSPM_6(x)      # B=1, C=512, H=16, W=16
        out = torch.cat([x, LSPM_1, LSPM_2, LSPM_3, LSPM_6], dim=1) # B=1, C=2560, H=16, W=16
        out = self.conv(out)         # B=1, C=512, H=16, W=16
        return out

class FAMCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FAMCA, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Conv2d(3*out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_cat = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels//4, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, left, down, right):
        # left1,    down 2,   right 3

        down = F.interpolate(down, size=left.size()[2], mode='bilinear', align_corners=True)
        down = self.relu(self.conv(down))

        merge = torch.cat((left, down, right), dim=1)  #1 1536 32 32
        merge = self.relu_cat(self.conv_cat(merge))    #1 512 32 32

        b, c, _, _ = merge.size()                      #1 512 32 32
        y = self.avg_pool(merge).view(b, c)            #1 512
        y = self.fc(y).view(b, c, 1, 1)                #1 512 1 1

        out = torch.mul(y, merge)                      #1 512 32 32

        return out

class FAMCA_Single(nn.Module):
    def __init__(self, channels):
        super(FAMCA_Single, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3*channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//4, channels, bias=False),
            nn.Sigmoid()
        )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.conv(self.upsample_2(x)))

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = torch.mul(y, x)

        return out

class model_VGG(nn.Module):
    def __init__(self, channel=32):
        super(model_VGG, self).__init__()
        self.vgg = VGG()

        self.score = nn.Conv2d(128, 1, 1, 1)

        self.lspm = LSPM(512)

        self.aggregation_4 = FAMCA(512, 512)
        self.aggregation_3 = FAMCA(512, 256)
        self.aggregation_2 = FAMCA(256, 128)
        self.aggregation_1 = FAMCA_Single(128)

        self.out_planes = [512, 256, 128]
        infos = []
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(512, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):

        x1 = self.vgg.conv1(x)   #1 64 256 256
        x2 = self.vgg.conv2(x1)  #1 128 128 128
        x3 = self.vgg.conv3(x2)  #1 256 64 64
        x4 = self.vgg.conv4(x3)  #1 512 32 32
        x5 = self.vgg.conv5(x4)  #1 512 16 16

        lspm = self.lspm(x5)     #1 512 16 16
        GG = []
        GG.append(self.infos[0](self.upsample_2(lspm))) #1 512 32 32
        GG.append(self.infos[1](self.upsample_4(lspm))) #1 256 64 64
        GG.append(self.infos[2](self.upsample_8(lspm))) #1 128 128 128

        merge  = self.aggregation_4(x4, x5,    GG[0])
        merge  = self.aggregation_3(x3, merge, GG[1])
        merge  = self.aggregation_2(x2, merge, GG[2])
        merge  = self.aggregation_1(merge)
        merge  = self.score(merge)                      #1 1 256 256
        result = F.interpolate(merge, x1.size()[2], mode='bilinear', align_corners=True)   #1 1 256 256

        return result

if __name__ == "__main__":
    import torch as t

    rgb = t.randn(1, 3, 256, 256)

    net = model_VGG()

    out = net(rgb)

    print(out.shape)

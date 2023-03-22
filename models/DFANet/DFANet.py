import torch
import torch.nn as nn
import torch.nn.functional as F

# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, inputChannel, outputChannel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inputChannel, inputChannel, kernel_size, stride, padding, dilation,
                               groups=inputChannel, bias=bias)
        self.pointwise = nn.Conv2d(inputChannel, outputChannel, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

# Xception子模块
class Block(nn.Module):
    def __init__(self, inputChannel, outputChannel, stride=1, BatchNorm=nn.BatchNorm2d):
        super(Block, self).__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(inputChannel, outputChannel // 4),
                                   BatchNorm(outputChannel // 4),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(SeparableConv2d(outputChannel // 4, outputChannel // 4),
                                   BatchNorm(outputChannel // 4),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(SeparableConv2d(outputChannel // 4, outputChannel, stride=stride),
                                   BatchNorm(outputChannel),
                                   nn.ReLU())
        self.projection = nn.Conv2d(inputChannel, outputChannel, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.projection(x)
        return out + identity


class enc(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeat=4):
        super(enc, self).__init__()
        stacks = [Block(in_channels, out_channels, stride=2)]
        for x in range(num_repeat - 1):
            stacks.append(Block(out_channels, out_channels))
        self.build = nn.Sequential(*stacks)

    def forward(self, x):
        x = self.build(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True))

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class SubBranch(nn.Module):
    def __init__(self, channel_cfg, branch_index):
        super(SubBranch, self).__init__()
        self.enc2 = enc(channel_cfg[0], 48, num_repeat=4)
        self.enc3 = enc(channel_cfg[1], 96, num_repeat=6)
        self.enc4 = enc(channel_cfg[2], 192, num_repeat=4)
        self.atten = Attention(192)
        self.branch_index = branch_index

    def forward(self, x0, *args):
        out0 = self.enc2(x0)
        if self.branch_index in [1, 2]:
            out1 = self.enc3(torch.cat([out0, args[0]], 1))
            out2 = self.enc4(torch.cat([out1, args[1]], 1))
        else:
            out1 = self.enc3(out0)
            out2 = self.enc4(out1)
        out3 = self.atten(out2)
        return [out0, out1, out2, out3]


class DFA_Encoder(nn.Module):
    def __init__(self, channel_cfg):
        super(DFA_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU())
        self.branch0 = SubBranch(channel_cfg[0], branch_index=0)
        self.branch1 = SubBranch(channel_cfg[1], branch_index=1)
        self.branch2 = SubBranch(channel_cfg[2], branch_index=2)

    def forward(self, x):
        x = self.conv1(x)

        x0, x1, x2, x5 = self.branch0(x)
        x3 = F.interpolate(x5, x0.size()[2:], mode='bilinear', align_corners=True)
        x1, x2, x3, x6 = self.branch1(torch.cat([x0, x3], 1), x1, x2)
        x4 = F.interpolate(x6, x1.size()[2:], mode='bilinear', align_corners=True)
        x2, x3, x4, x7 = self.branch2(torch.cat([x1, x4], 1), x2, x3)

        return [x0, x1, x2, x5, x6, x7]


class DFA_Decoder(nn.Module):
    """
        Decoder of DFANet.
    """

    def __init__(self, decode_channels, num_classes):
        super(DFA_Decoder, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(decode_channels),
                                   nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(decode_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(decode_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(decode_channels),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(decode_channels),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(decode_channels),
                                   nn.ReLU(inplace=True))

        self.conv_add1 = nn.Sequential(
            nn.Conv2d(in_channels=decode_channels, out_channels=decode_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True))

        self.conv_cls = nn.Conv2d(in_channels=decode_channels, out_channels=num_classes, kernel_size=3, padding=1,
                                  bias=False)

    def forward(self, x0, x1, x2, x3, x4, x5):
        x0 = self.conv0(x0)
        x1 = F.interpolate(self.conv1(x1), x0.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.conv2(x2), x0.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.conv3(x3), x0.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.conv5(x4), x0.size()[2:], mode='bilinear', align_corners=True)
        x5 = F.interpolate(self.conv5(x5), x0.size()[2:], mode='bilinear', align_corners=True)

        x_shallow = self.conv_add1(x0 + x1 + x2)

        x = self.conv_cls(x_shallow + x3 + x4 + x5)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x


class DFANet(nn.Module):
    def __init__(self, channel_cfg, decoder_channel=64, num_classes=1):
        super(DFANet, self).__init__()
        self.encoder = DFA_Encoder(channel_cfg)
        self.decoder = DFA_Decoder(decoder_channel, num_classes)

    def forward(self, x):
        x0, x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x0, x1, x2, x3, x4, x5)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ch_cfg = [[8, 48, 96],
              [240, 144, 288],
              [240, 144, 288]]

    model = DFANet(ch_cfg, 64, 1)
    model.eval()
    image = torch.randn(1, 3, 256, 256)
    out = model(image)
    print(out.shape)

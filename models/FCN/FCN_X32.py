# encoding: utf-8
"""补充内容见model and loss.ipynb & 自定义双向线性插值滤子（卷积核）.ipynb"""

import numpy as np
import torch
from torchvision import models
from torch import nn
import os


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# path_state_dict = os.path.join(BASE_DIR, "..", "vgg16_bn-6c64b313.pth")
path_state_dict = os.path.join('E:/03-博士/学术研究/03红外弱小目标探测编程/Small_detection_methods/AGPCNet(ALL_Methods)/models/FCN/vgg16_bn-6c64b313.pth')
def get_vgg16(path_state_dict):
    model = models.vgg16_bn()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    return model
pretrained_net = get_vgg16(path_state_dict)


# pretrained_net = models.vgg16_bn(pretrained=False)
# print(pretrained_net.features)

class FCN_X32(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.stage1 = pretrained_net.features[:7] #第一个池化
        self.stage2 = pretrained_net.features[7:14] #第二个池化
        self.stage3 = pretrained_net.features[14:24] #第三个池化
        self.stage4 = pretrained_net.features[24:34] #第四个池化
        self.stage5 = pretrained_net.features[34:] #第五个池化

        self.scores1 = nn.Conv2d(512, num_classes, 1) #把通道数变为num_classes
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernel(512, 512, 4)

        self.upsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernel(256, 256, 4)

    def forward(self, x):
        s1 = self.stage1(x)   #1 64 128 128
        s2 = self.stage2(s1)  #1 128 64 64
        s3 = self.stage3(s2)  #1 256 32 32
        s4 = self.stage4(s3)  #1 512 16 16
        s5 = self.stage5(s4)  #1 512 8 8

        # scores1 = self.scores1(s5) #通道数num_classes 12
        s5 = self.upsample_2x_1(s5) #1 512 16 16
        add1 = s5 + s4              #1 512 16 16

        # scores2 = self.scores2(add1) #通道数num_classes 12
        add1 = self.conv_trans1(add1)   #1 256 16 16
        add1 = self.upsample_2x_2(add1) #1 256 32 32
        add2 = add1 + s3                #1 256 32 32

        output = self.conv_trans2(add2) #1 1 32 32
        output = self.upsample_8x(output) #1 1 256 256
        return output

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)



if __name__ == "__main__":
    import torch as t
    print('-----'*5)
    rgb = t.randn(1, 3, 256, 256)
    net = FCN_X32()
    out = net(rgb)
    print(out.shape)
    # param_size, param_sum, buffer_size, buffer_sum, all_size = getModelSize(FCN_X32(rgb))
    # print(all_size)

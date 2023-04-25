import os
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from public.backbone import DeformableConv2d


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CenterNetNeck(nn.Module):
    def __init__(self,
                 inplanes: list,
                 num_layers: int = 3,
                 out_channels: list = [256, 128, 64],
                 upsample=False,
                 depthwise=False):
        """
        @description  :normal CenterNet Neck
        ---------
        @param  :
        inplanes: the channel_num of the stage3 to stage5 in backbone(for example, [512,1024,2048] in resnet50)
        num_layers: the upsample layers num in neck, set 3 as default
        out_channels: the out_channle_num in per layer
        -------
        @Returns  :
        one featuremap with 8x upsample ratio to the input, channel_num = out_channels[-1]
        -------
        """

        super(CenterNetNeck, self).__init__()

        Conv = DWConv if depthwise else BaseConv
        self.inplanes = inplanes[-1]
        layers = []
        for i in range(num_layers):
            layers.append(Conv(self.inplanes, out_channels[i], 3, 1))

            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            else:
                layers.append(
                    nn.ConvTranspose2d(in_channels=out_channels[i],
                                       out_channels=out_channels[i],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       output_padding=0,
                                       bias=False)
                )
                layers.append(nn.BatchNorm2d(out_channels[i]))
                layers.append(nn.ReLU(inplace=True))

            self.inplanes = out_channels[i]

        self.deconv_neck = nn.Sequential(*layers)

        for m in self.deconv_neck.modules():
            if isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                                1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x):
        out = self.deconv_neck(x[-1])
        del x
        return [out]

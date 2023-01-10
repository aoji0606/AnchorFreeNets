import os
import sys
import math

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch import Tensor


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SEBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1):
        super(SEBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()
        # self.stride = stride

    def forward(self, x):
        residual = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)

        out = self.ca(x) * x
        out = self.sa(out) * out

        out += residual
        out = self.relu(out)

        return out


class UP_layer(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, deformable=False, depthwise=False):
        super(UP_layer, self).__init__()
        if deformable:
            from public.backbone import DeformableConv2d
            self.conv = DeformableConv2d(
                in_channels, out_channels, kernel_size=(3, 3),
                stride=1, padding=1, groups=1, bias=False
            )
        else:
            if depthwise:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                              groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if upsample:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)  # onnx does not support this method
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=out_channels, out_channels=out_channels, kernel_size=4,
                    stride=2, padding=1, output_padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        for m in self.modules():
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
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.upsample(out)
        return out


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, selayer=False, depthwise=False):
        super(ShortCut, self).__init__()
        if selayer:
            self.conv1 = nn.Sequential(
                SEBlock(in_channels),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
                )
            )
        else:
            if depthwise:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                              groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                )
            else:
                self.conv1 = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
                )

        for m in self.conv1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.conv1(x)


class TTFNeck(nn.Module):
    def __init__(self, inplanes: list, out_channels: list = [256, 128, 64], upsample=True, selayer: bool = False,
                 deformable: bool = False, depthwise: bool = False):
        """
        @description  :CenterNet Neck in  TTF
        ---------
        @param  :
        inplanes: the channel_num of the stage3 to stage5 in backbone(for example, [512,1024,2048] in resnet50)
        out_channels: the out_channle_num in per layer
        selayer: use the CBAM attention
        -------
        @Returns  :
        one featuremap with 8x upsample ratio to the input, channel_num = out_channels[-1]
        -------
        """
        super(TTFNeck, self).__init__()
        self.up5t4 = UP_layer(inplanes[-1], out_channels[0], upsample, deformable, depthwise)
        self.up4t3 = UP_layer(out_channels[0], out_channels[1], upsample, deformable, depthwise)
        self.up3t2 = UP_layer(out_channels[1], out_channels[2], upsample, deformable, depthwise)

        self.shortcut4 = ShortCut(int(inplanes[-2]), out_channels[0], selayer, depthwise)
        self.shortcut3 = ShortCut(int(inplanes[-3]), out_channels[1], selayer, depthwise)
        self.shortcut2 = ShortCut(int(inplanes[-4]), out_channels[2], selayer, depthwise)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        C2, C3, C4, C5 = x
        C4_up = self.up5t4(C5) + F.relu(self.shortcut4(C4))
        C3_up = self.up4t3(C4_up) + F.relu(self.shortcut3(C3))
        C2_up = self.up3t2(C3_up) + self.shortcut2(C2)
        return [C2_up]


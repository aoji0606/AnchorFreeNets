import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import c2_xavier_fill
import math


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 groups=1,
                 has_bn=True,
                 has_act=True):
        super(ConvBnActBlock, self).__init__()
        self.has_bn = has_bn
        self.has_act = has_act
        self.conv = nn.Conv2d(inplanes,
                              planes,
                              kernel_size,
                              stride=stride,
                              padding=kernel_size // 2,
                              groups=groups,
                              bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(planes)
        if self.has_act:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_act:
            x = self.act(x)

        return x


class YOLOV3FPNHead(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 num_anchors=3,
                 num_classes=80):
        super(YOLOV3FPNHead, self).__init__()
        P5_1_layers = []
        for i in range(5):
            if i % 2 == 0:
                P5_1_layers.append(
                    ConvBnActBlock(C5_inplanes,
                                   C5_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            else:
                P5_1_layers.append(
                    ConvBnActBlock(C5_inplanes // 2,
                                   C5_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
        self.P5_1 = nn.Sequential(*P5_1_layers)
        self.P5_up_conv = ConvBnActBlock(C5_inplanes // 2,
                                         C4_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         has_bn=True,
                                         has_act=True)
        self.P5_2 = ConvBnActBlock(C5_inplanes // 2,
                                   C5_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)
        self.P5_pred_conv = nn.Conv2d(C5_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      bias=True)

        P4_1_layers = []
        for i in range(5):
            if i == 0:
                P4_1_layers.append(
                    ConvBnActBlock((C4_inplanes // 2) + C4_inplanes,
                                   C4_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 1:
                P4_1_layers.append(
                    ConvBnActBlock(C4_inplanes // 2,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 0:
                P4_1_layers.append(
                    ConvBnActBlock(C4_inplanes,
                                   C4_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
        self.P4_1 = nn.Sequential(*P4_1_layers)
        self.P4_up_conv = ConvBnActBlock(C4_inplanes // 2,
                                         C3_inplanes // 2,
                                         kernel_size=1,
                                         stride=1,
                                         has_bn=True,
                                         has_act=True)
        self.P4_2 = ConvBnActBlock(C4_inplanes // 2,
                                   C4_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)
        self.P4_pred_conv = nn.Conv2d(C4_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      bias=True)

        P3_1_layers = []
        for i in range(5):
            if i == 0:
                P3_1_layers.append(
                    ConvBnActBlock((C3_inplanes // 2) + C3_inplanes,
                                   C3_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 1:
                P3_1_layers.append(
                    ConvBnActBlock(C3_inplanes // 2,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
            elif i % 2 == 0:
                P3_1_layers.append(
                    ConvBnActBlock(C3_inplanes,
                                   C3_inplanes // 2,
                                   kernel_size=1,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True))
        self.P3_1 = nn.Sequential(*P3_1_layers)
        self.P3_2 = ConvBnActBlock(C3_inplanes // 2,
                                   C3_inplanes,
                                   kernel_size=3,
                                   stride=1,
                                   has_bn=True,
                                   has_act=True)
        self.P3_pred_conv = nn.Conv2d(C3_inplanes,
                                      num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      bias=True)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)

        del C5

        C5_upsample = self.P5_up_conv(P5)
        C5_upsample = F.interpolate(C5_upsample,
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='nearest')

        C4 = torch.cat([C4, C5_upsample], axis=1)
        del C5_upsample
        P4 = self.P4_1(C4)
        del C4
        C4_upsample = self.P4_up_conv(P4)
        C4_upsample = F.interpolate(C4_upsample,
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='nearest')

        C3 = torch.cat([C3, C4_upsample], axis=1)
        del C4_upsample
        P3 = self.P3_1(C3)
        del C3

        P5 = self.P5_2(P5)
        P5 = self.P5_pred_conv(P5)

        P4 = self.P4_2(P4)
        P4 = self.P4_pred_conv(P4)

        P3 = self.P3_2(P3)
        P3 = self.P3_pred_conv(P3)

        return [P3, P4, P5]
import os
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

# from public.detection.models.DCNv2 import DCN



class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def hard_swish(self, x, inplace):
        inner = F.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if inplace else x.mul(inner)

    def forward(self, x):
        return self.hard_swish(x, self.inplace)


class SeparableConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(inplanes,
                                        inplanes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=inplanes,
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(inplanes,
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class EfficientDetClsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors,
                 num_classes,
                 num_layers,
                 prior=0.01):
        super(EfficientDetClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(SeparableConvBlock(inplanes, inplanes))
            layers.append(nn.BatchNorm2d(inplanes))
            layers.append(HardSwish(inplace=True))

        layers.append(SeparableConvBlock(
            inplanes,
            num_anchors * num_classes,
        ))
        self.cls_head = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_head[-1].pointwise_conv.bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
        x = self.sigmoid(x)

        return x


class EfficientDetRegHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_layers):
        super(EfficientDetRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(SeparableConvBlock(inplanes, inplanes))
            layers.append(nn.BatchNorm2d(inplanes))
            layers.append(HardSwish(inplace=True))

        layers.append(SeparableConvBlock(
            inplanes,
            num_anchors * 4,
        ))
        self.reg_head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.reg_head(x)

        return x
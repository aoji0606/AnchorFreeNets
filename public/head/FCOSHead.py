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



class FCOSHead(nn.Module):
    def __init__(self,
                 inplanes:int,
                 num_classes:int,
                 num_layers:int=4,
                 prior:float=0.01,
                 use_gn:bool=True,
                 cnt_on_reg:bool=True):
        super(FCOSHead, self).__init__()
        self.cnt_on_reg = cnt_on_reg

        cls_layers = []
        for _ in range(num_layers):
            cls_layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            if use_gn:
                cls_layers.append(nn.GroupNorm(32, inplanes))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*cls_layers)

        reg_layers = []
        for _ in range(num_layers):
            reg_layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            if use_gn:
                reg_layers.append(nn.GroupNorm(32, inplanes))
            reg_layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*reg_layers)

        self.cls_out = nn.Conv2d(inplanes,
                                 num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.reg_out = nn.Conv2d(inplanes,
                                 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.center_out = nn.Conv2d(inplanes,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, x):
        cls_x = self.cls_head(x)
        reg_x = self.reg_head(x)

        del x

        cls_output = self.cls_out(cls_x)
        reg_output = self.reg_out(reg_x)

        if self.cnt_on_reg:
            center_output = self.center_out(reg_x)
        else:
            center_output = self.center_out(cls_x)

        return cls_output, reg_output, center_output
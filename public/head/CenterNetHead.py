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

# from public.head.DCNv2 import DCN


class CenterNetHead(nn.Module):
    def __init__(self,
                 num_classes:int,
                 out_channels:int=64):
        """
        @description  :
        ---------
        @param  :
        num_classes: the class nums in dataset
        out_channels: the out channels in neck
        -------
        @Returns  :
        -------
        """
        
        
        super(CenterNetHead, self).__init__()

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(out_channels,
                      64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,
                      num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(out_channels,
                      64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(out_channels,
                      64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )

        self.heatmap_head[-1].bias.data.fill_(-2.19)

        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        heatmap_output = self.heatmap_head(x[-1])
        offset_output = self.offset_head(x[-1])
        wh_output = self.wh_head(x[-1])
        del x

        return heatmap_output, offset_output, wh_output
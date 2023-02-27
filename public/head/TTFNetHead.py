import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
sys.path.append(BASE_DIR)


class TTFNetHead(nn.Module):
    def __init__(self, num_classes: int, out_channels: int = 64, wh_offset_base=16, depthwise=False):
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
        super(TTFNetHead, self).__init__()

        self.wh_offset_base = wh_offset_base

        if depthwise:
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                          groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True),
            )
            self.wh_head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                          groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 4, kernel_size=1, stride=1, bias=True),
            )
        else:
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(out_channels, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True),
            )
            self.wh_head = nn.Sequential(
                nn.Conv2d(out_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 4, kernel_size=1, stride=1, bias=True),
            )

        self.init_weight()

    def init_weight(self):
        for m in self.heatmap_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    pass
        nn.init.constant_(self.heatmap_head[-1].bias, float(-np.log((1 - 0.01) / 0.01)))

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    pass

    def forward(self, x):
        heatmap_output = self.heatmap_head(x[-1])
        wh_output = F.relu(self.wh_head(x[-1])) * self.wh_offset_base  # scale wh
        return heatmap_output, wh_output


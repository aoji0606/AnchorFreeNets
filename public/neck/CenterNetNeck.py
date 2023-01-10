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

class CenterNetNeck(nn.Module):
    def __init__(self,
                 inplanes:list,
                 num_layers:int=3,
                 out_channels:list=[256, 128, 64],
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
        self.inplanes = inplanes[-1]
        layers = []
        for i in range(num_layers):
            if depthwise:
                layers.append(
                    nn.Conv2d(in_channels=self.inplanes,
                        out_channels=self.inplanes,
                        kernel_size=(3, 3),
                        stride=1,
                        padding=1,
                        groups=self.inplanes,
                        bias=False),
                )
                layers.append(nn.BatchNorm2d(self.inplanes))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Conv2d(self.inplanes, out_channels[i], kernel_size=1, stride=1))
            else:
                layers.append(
#                     DCN(in_channels=self.inplanes,
#                         out_channels=out_channels[i],
#                         kernel_size=(3, 3),
#                         stride=1,
#                         padding=1,
#                         dilation=1,
#                         deformable_groups=1))
#                     DeformableConv2d(in_channels=self.inplanes,
#                                     out_channels=out_channels[i],
#                                     kernel_size=(3, 3),
#                                     stride=1,
#                                     padding=1,
#                                     groups=1,
#                                     bias=False)
                    nn.Conv2d(in_channels=self.inplanes,
                        out_channels=out_channels[i],
                        kernel_size=(3, 3),
                        stride=1,
                        padding=1)
                    )
            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.ConvTranspose2d(in_channels=out_channels[i],
                                   out_channels=out_channels[i],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   output_padding=0,
                                   bias=False))
            layers.append(nn.BatchNorm2d(out_channels[i]))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = out_channels[i]

        self.public_deconv_head = nn.Sequential(*layers)

        for m in self.public_deconv_head.modules():
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
        out = self.public_deconv_head(x[-1])
        del x
        return [out]
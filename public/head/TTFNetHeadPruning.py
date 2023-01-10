import os
import sys
import numpy as np

import torch
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


class TTFNetHeadPruning(nn.Module):
    def __init__(self, num_classes:int, out_channels:int=64, wh_offset_base=16):
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
        super(TTFNetHeadPruning, self).__init__()

        self.wh_offset_base = wh_offset_base
        self.num_classes = num_classes
        self.out_channels = out_channels

        self.heatmap_head = nn.Sequential()
        self.heatmap_head.add_module('conv1', nn.Conv2d(out_channels, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.heatmap_head.add_module('bn1', nn.BatchNorm2d(128))
        self.heatmap_head.add_module('mask1', nn.Conv2d(128, 128, 1, groups=128, bias=False))
        self.heatmap_head.add_module('relu1', nn.ReLU(inplace=True))
        self.heatmap_head.add_module('conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True))
        self.heatmap_head.add_module('bn2', nn.BatchNorm2d(128))
        self.heatmap_head.add_module('mask2', nn.Conv2d(128, 128, 1, groups=128, bias=False))
        self.heatmap_head.add_module('relu2', nn.ReLU(inplace=True))
        self.heatmap_head.add_module('conv3', nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.wh_head = nn.Sequential()
        self.wh_head.add_module('conv1', nn.Conv2d(out_channels, 64, kernel_size=3, stride=1, padding=1, bias=True))
        self.wh_head.add_module('bn1', nn.BatchNorm2d(64))
        self.wh_head.add_module('mask1', nn.Conv2d(64, 64, 1, groups=64, bias=False))
        self.wh_head.add_module('relu1', nn.ReLU(inplace=True))
        self.wh_head.add_module('conv2', nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0, bias=True))

        # self.heatmap_head = nn.Sequential(
        #     nn.Conv2d(out_channels, 128, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        # )
        # self.wh_head = nn.Sequential(
        #     nn.Conv2d(out_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0, bias=True),
        # )

        self.init_weight()
        nn.init.ones_(self.heatmap_head.mask1.weight)
        nn.init.ones_(self.heatmap_head.mask2.weight)
        nn.init.ones_(self.wh_head.mask1.weight)
    
    def init_weight(self):
        for m in self.heatmap_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.heatmap_head[-1].bias, float(-np.log((1 - 0.01) / 0.01)))

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        heatmap_output = self.heatmap_head(x[-1])
        wh_output = F.relu(self.wh_head(x[-1]) + 0.1) * self.wh_offset_base  # scale wh
        return heatmap_output, wh_output
    
    def prune_by_mask_and_thresh(self, in_masks, thresh=0, out_mask=None):

        def prune_conv_bn_mask(conv, bn, mask, in_mask, out_mask):
            temp_conv = nn.Conv2d(
                int(in_mask.sum()),
                int(out_mask.sum()),
                conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=False
            )
            temp_conv.weight.data = conv.weight.data[out_mask][:, in_mask]

            temp_bn = nn.BatchNorm2d(int(out_mask.sum()))
            temp_bn.weight.data = bn.weight.data[out_mask]
            temp_bn.bias.data = bn.bias.data[out_mask]
            temp_bn.running_mean = bn.running_mean[out_mask]
            temp_bn.running_var = bn.running_var[out_mask]

            temp_mask = nn.Conv2d(
                int(out_mask.sum()),
                int(out_mask.sum()),
                mask.kernel_size,
                stride=mask.stride,
                padding=mask.padding,
                groups=int(out_mask.sum()),
                bias=False
            )
            temp_mask.weight.data = mask.weight.data[out_mask]

            return temp_conv, temp_bn, temp_mask

        ret_masks = []

        prune_cnt = 0; total_cnt = 0

        # prune headmap head
        in_mask = in_masks[0]
        mask = self.heatmap_head.mask1.weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        temp_conv, temp_bn, temp_mask = prune_conv_bn_mask(
            self.heatmap_head.conv1, self.heatmap_head.bn1, self.heatmap_head.mask1,
            in_mask, mask # self.heatmap_head.mask1.weight.data.reshape(-1)[mask]
        )
        self.heatmap_head.conv1 = temp_conv
        self.heatmap_head.bn1 = temp_bn
        self.heatmap_head.mask1 = temp_mask
        # self.heatmap_head.__delattr__('mask1')
        
        in_mask = mask
        mask = self.heatmap_head.mask2.weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        temp_conv, temp_bn, temp_mask = prune_conv_bn_mask(
            self.heatmap_head.conv2, self.heatmap_head.bn2, self.heatmap_head.mask2,
            in_mask, mask # self.heatmap_head.mask2.weight.data.reshape(-1)[mask]
        )
        self.heatmap_head.conv2 = temp_conv
        self.heatmap_head.bn2 = temp_bn
        self.heatmap_head.mask2 = temp_mask
        # self.heatmap_head.__delattr__('mask2')

        in_mask = mask
        temp_conv = nn.Conv2d(
            int(in_mask.sum()),
            self.num_classes,
            self.heatmap_head.conv3.kernel_size,
            stride=self.heatmap_head.conv3.stride,
            padding=self.heatmap_head.conv3.padding,
            bias=self.heatmap_head.conv3.bias is not None
        ).eval()
        temp_conv.weight.data = self.heatmap_head.conv3.weight.data[:, in_mask]
        if hasattr(self.heatmap_head.conv3, 'bias') and self.heatmap_head.conv3.bias is not None:
            temp_conv.bias.data = self.heatmap_head.conv3.bias.data
        self.heatmap_head.conv3 = temp_conv

        # prune wh head
        in_mask = in_masks[0]
        mask = self.wh_head.mask1.weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        temp_conv, temp_bn, temp_mask = prune_conv_bn_mask(
            self.wh_head.conv1, self.wh_head.bn1, self.wh_head.mask1,
            in_mask, mask # , self.wh_head.mask1.weight.data.reshape(-1)[mask]
        )
        self.wh_head.conv1 = temp_conv
        self.wh_head.bn1 = temp_bn
        self.wh_head.mask1 = temp_mask
        # self.wh_head.__delattr__('mask1')

        in_mask = mask
        temp_conv = nn.Conv2d(
            int(in_mask.sum()),
            4,
            self.wh_head.conv2.kernel_size,
            stride=self.wh_head.conv2.stride,
            padding=self.wh_head.conv2.padding,
            bias=self.wh_head.conv2.bias is not None
        ).eval()
        temp_conv.weight.data = self.wh_head.conv2.weight.data[:, in_mask]
        if hasattr(self.wh_head.conv2, 'bias') and self.wh_head.conv2.bias is not None:
            temp_conv.bias.data = self.wh_head.conv2.bias.data
        self.wh_head.conv2 = temp_conv
        
        return ret_masks, prune_cnt, total_cnt
    
    def merge_masks(self):
        # heatmap head
        self.heatmap_head.bn1.weight.data *= self.heatmap_head.mask1.weight.data.reshape(-1)
        self.heatmap_head.bn1.bias.data *= self.heatmap_head.mask1.weight.data.reshape(-1)
        self.heatmap_head.__delattr__('mask1')

        self.heatmap_head.bn2.weight.data *= self.heatmap_head.mask2.weight.data.reshape(-1)
        self.heatmap_head.bn2.bias.data *= self.heatmap_head.mask2.weight.data.reshape(-1)
        self.heatmap_head.__delattr__('mask2')

        # wh head
        self.wh_head.bn1.weight.data *= self.wh_head.mask1.weight.data.reshape(-1)
        self.wh_head.bn1.bias.data *= self.wh_head.mask1.weight.data.reshape(-1)
        self.wh_head.__delattr__('mask1')

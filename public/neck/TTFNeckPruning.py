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

from public.backbone import DeformableConv2d


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
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.ca(x) * x
        out = self.sa(out) * out
        out = self.relu(out + residual)
        return out


class UPLayerPruning(nn.Module):
    def __init__(self, in_channels, out_channels, deformable=True):
        super(UPLayerPruning, self ).__init__()
        
        self.deformable = deformable

        if deformable:
            self.conv = DeformableConv2d(
                in_channels, out_channels, kernel_size=(3, 3),
                stride=1, padding=1, groups=1, bias=False
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 3),
                stride=1, padding=1, bias=False
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.mask = nn.Conv2d(out_channels, out_channels, 1, groups=out_channels, bias=False)
        self.relu = nn.ReLU()

        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)  # onnx does not support this method
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
        
        nn.init.ones_(self.mask.weight)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if hasattr(self, 'mask'):
            out = self.mask(out)
        out = self.relu(out)
        out = self.upsample(out)
        return out
    
    def merge_masks(self):
        self.bn.weight.data *= self.mask.weight.data.reshape(-1)
        self.bn.bias.data *= self.mask.weight.data.reshape(-1)
        self.__delattr__('mask')
    
    def prune_by_mask_and_thresh(self, in_mask, thresh=0, out_mask_tensor=None):
        prune_cnt = 0; total_cnt = 0

        # prune conv
        mask = self.mask.weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        if self.deformable:
            temp_conv = DeformableConv2d(
                int(in_mask.sum()), int(mask.sum()), kernel_size=(3, 3),
                stride=1, padding=1, groups=1, bias=False
            ).eval()
            temp_conv.offset_conv.weight.data = self.conv.offset_conv.weight.data[:, in_mask]
            temp_conv.offset_conv.bias.data = self.conv.offset_conv.bias.data
            temp_conv.mask_conv.weight.data = self.conv.mask_conv.weight.data[:, in_mask]
            temp_conv.mask_conv.bias.data = self.conv.mask_conv.bias.data
        else:
            temp_conv = nn.Conv2d(
                int(in_mask.sum()), int(mask.sum()), kernel_size=(3, 3),
                stride=1, padding=1, bias=False
            ).eval()
        temp_conv.weight.data = self.conv.weight.data[mask][:, in_mask]
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            temp_conv.bias.data = self.conv.bias.data[mask]

        self.conv = temp_conv

        # prune bn
        temp_bn = nn.BatchNorm2d(int(mask.sum())).eval()
        temp_bn.weight.data = self.bn.weight.data[mask]
        temp_bn.bias.data = self.bn.bias.data[mask]
        temp_bn.running_mean = self.bn.running_mean[mask]
        temp_bn.running_var = self.bn.running_var[mask]
        self.bn = temp_bn
        
        # del mask
        # self.bn.weight.data *= self.mask.weight.data.reshape(-1)[mask]
        # self.bn.bias.data *= self.mask.weight.data.reshape(-1)[mask]
        # self.__delattr__('mask')
        temp_mask = nn.Conv2d(
                int(mask.sum()),
                int(mask.sum()),
                self.mask.kernel_size,
                stride=self.mask.stride,
                padding=self.mask.padding,
                groups=int(mask.sum()),
                bias=False
            )
        temp_mask.weight.data = self.mask.weight.data[mask]
        self.mask = temp_mask

        # prune upsample
        assert out_mask_tensor is not None, 'out_mask_tensor must not None.'
        in_mask = mask
        mask = out_mask_tensor > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        temp_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=int(in_mask.sum()), out_channels=int(mask.sum()), kernel_size=4,
                stride=2, padding=1, output_padding=0, bias=False
            ),
            nn.BatchNorm2d(int(mask.sum())),
            nn.ReLU(inplace=True)
        )

        temp_upsample[0].weight.data = self.upsample[0].weight.data[in_mask][:, mask] # deconv: [in, out, k, k]
        if hasattr(self.upsample[0], 'bias') and self.upsample[0].bias is not None:
            temp_upsample[0].bias.data = self.upsample[0].bias.data[mask]

        temp_upsample[1].weight.data = self.upsample[1].weight.data[mask]
        temp_upsample[1].bias.data = self.upsample[1].bias.data[mask]
        temp_upsample[1].running_mean = self.upsample[1].running_mean[mask]
        temp_upsample[1].running_var = self.upsample[1].running_var[mask]

        # temp_upsample[1].weight.data *= out_mask_tensor[mask]
        # temp_upsample[1].bias.data *= out_mask_tensor[mask]

        self.upsample = temp_upsample

        return [], prune_cnt, total_cnt


class ShortCutPruning(nn.Module):
    def __init__(self, in_channels, out_channels, selayer=False):
        super(ShortCutPruning, self).__init__()
        
        self.selayer = selayer

        if selayer:
            self.conv1 = nn.Sequential(
                SEBlock(in_channels),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
                )
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
    
    def prune_by_mask_and_thresh(self, in_mask, thresh=0, out_mask_tensor=None):
        assert out_mask_tensor is not None, 'out_mask_tensor must not None.'
        
        prune_cnt, total_cnt = 0, 0

        mask = out_mask_tensor > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        if self.selayer:
            raise NotImplementedError('Pruning of the selayer needs to be implemented.')
        else:
            temp_conv = nn.Conv2d(
                int(in_mask.sum()), int(mask.sum()), kernel_size=(3, 3),
                stride=1, padding=1, bias=False
            ).eval()
            temp_conv.weight.data = self.conv1.weight.data[mask][:, in_mask]
            if hasattr(self.conv1, 'bias') and self.conv1.bias is not None:
                temp_conv.bias.data = self.conv1.bias.data[mask]

        # temp_conv.weight.data *= out_mask_tensor[mask].reshape(-1, 1, 1, 1)
        # if hasattr( self.conv1, 'bias') and  self.conv1.bias is not None:
        #     temp_conv.bias.data *= out_mask_tensor[mask]
        
        self.conv1 = temp_conv

        return [], prune_cnt, total_cnt


class TTFNeckPruning(nn.Module):
    def __init__(self, inplanes:list, out_channels:list=[256, 128, 64], selayer:bool=False, deformable:bool=True):
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
        super(TTFNeckPruning, self).__init__()
        
        self.inplanes = inplanes
        self.out_channels = out_channels
        self.deformable = deformable
        self.selayer = selayer

        self.up5t4 = UPLayerPruning(inplanes[-1], out_channels[0], deformable)
        self.up4t3 = UPLayerPruning(out_channels[0], out_channels[1], deformable)
        self.up3t2 = UPLayerPruning(out_channels[1], out_channels[2], deformable)
        
        self.shortcut4 = ShortCutPruning(int(inplanes[-2]), out_channels[0], selayer)
        self.shortcut3 = ShortCutPruning(int(inplanes[-3]), out_channels[1], selayer)
        self.shortcut2 = ShortCutPruning(int(inplanes[-4]), out_channels[2], selayer)

        self.mask4 = nn.Conv2d(out_channels[0], out_channels[0], 1, groups=out_channels[0], bias=False)
        self.mask3 = nn.Conv2d(out_channels[1], out_channels[1], 1, groups=out_channels[1], bias=False)
        self.mask2 = nn.Conv2d(out_channels[2], out_channels[2], 1, groups=out_channels[2], bias=False)

        nn.init.ones_(self.mask4.weight)
        nn.init.ones_(self.mask3.weight)
        nn.init.ones_(self.mask2.weight)

    def forward(self, x:List[Tensor]) -> List[Tensor]:
        C2, C3, C4, C5 = x

        C4_up = self.up5t4(C5   ) + F.relu(self.shortcut4(C4))
        if hasattr(self, 'mask4'):
            C4_up = self.mask4(C4_up)

        C3_up = self.up4t3(C4_up) + F.relu(self.shortcut3(C3))
        if hasattr(self, 'mask3'):
            C3_up = self.mask3(C3_up)

        C2_up = self.up3t2(C3_up) + self.shortcut2(C2)
        if hasattr(self, 'mask2'):
            C2_up = self.mask2(C2_up)

        return [C2_up]

    def prune_by_mask_and_thresh(self, in_masks, thresh=0):
        ret_masks = [
            self.mask2.weight.data.abs().reshape(-1) > thresh
        ]

        prune_cnt = 0; total_cnt = 0

        # c4_up
        _m = self.mask4.weight.data.abs().reshape(-1)
        mask = _m > thresh
        _, _prune_cnt, _total_cnt = self.up5t4.prune_by_mask_and_thresh(in_masks[-1], thresh=thresh, out_mask_tensor=_m)
        prune_cnt += _prune_cnt; total_cnt += _total_cnt
        _, _prune_cnt, _total_cnt = self.shortcut4.prune_by_mask_and_thresh(in_masks[-2], thresh=thresh, out_mask_tensor=_m)
        prune_cnt += _prune_cnt; total_cnt += _total_cnt

        temp_mask = nn.Conv2d(
            int(mask.sum()),
            int(mask.sum()),
            self.mask4.kernel_size,
            stride=self.mask4.stride,
            padding=self.mask4.padding,
            groups=int(mask.sum()),
            bias=False
        )
        temp_mask.weight.data = self.mask4.weight.data[mask]
        self.mask4 = temp_mask

        # c3_up
        in_mask = mask
        _m = self.mask3.weight.data.abs().reshape(-1)
        mask = _m > thresh

        _, _prune_cnt, _total_cnt = self.up4t3.prune_by_mask_and_thresh(in_mask, thresh=thresh, out_mask_tensor=_m)
        prune_cnt += _prune_cnt; total_cnt += _total_cnt
        _, _prune_cnt, _total_cnt = self.shortcut3.prune_by_mask_and_thresh(in_masks[-3], thresh=thresh, out_mask_tensor=_m)
        prune_cnt += _prune_cnt; total_cnt += _total_cnt
        
        temp_mask = nn.Conv2d(
            int(mask.sum()),
            int(mask.sum()),
            self.mask3.kernel_size,
            stride=self.mask3.stride,
            padding=self.mask3.padding,
            groups=int(mask.sum()),
            bias=False
        )
        temp_mask.weight.data = self.mask3.weight.data[mask]
        self.mask3 = temp_mask
        
        # c2_up
        in_mask = mask
        _m = self.mask2.weight.data.abs().reshape(-1)
        mask = _m > thresh

        _, _prune_cnt, _total_cnt = self.up3t2.prune_by_mask_and_thresh(in_mask, thresh=thresh, out_mask_tensor=_m)
        prune_cnt += _prune_cnt; total_cnt += _total_cnt
        _, _prune_cnt, _total_cnt = self.shortcut2.prune_by_mask_and_thresh(in_masks[-4], thresh=thresh, out_mask_tensor=_m)
        prune_cnt += _prune_cnt; total_cnt += _total_cnt
        
        temp_mask = nn.Conv2d(
            int(mask.sum()),
            int(mask.sum()),
            self.mask2.kernel_size,
            stride=self.mask2.stride,
            padding=self.mask2.padding,
            groups=int(mask.sum()),
            bias=False
        )
        temp_mask.weight.data = self.mask2.weight.data[mask]
        self.mask2 = temp_mask

        return ret_masks, prune_cnt, total_cnt
    
    def merge_masks(self):

        def _merge_layer_mask(uplayer, shortcut, mask):
            uplayer.merge_masks()
            uplayer.upsample[1].weight.data *= mask.weight.data.reshape(-1)
            uplayer.upsample[1].bias.data *= mask.weight.data.reshape(-1)

            shortcut.conv1.weight.data *= mask.weight.data.reshape(-1, 1, 1, 1)
            if hasattr(shortcut.conv1, 'bias') and shortcut.conv1.bias is not None:
                shortcut.conv1.bias.data *= mask.weight.data.reshape(-1)
        
        _merge_layer_mask(self.up5t4, self.shortcut4, self.mask4)  # c4_up
        _merge_layer_mask(self.up4t3, self.shortcut3, self.mask3)  # c3_up
        _merge_layer_mask(self.up3t2, self.shortcut2, self.mask2)  # c2_up

        self.__delattr__('mask4')
        self.__delattr__('mask3')
        self.__delattr__('mask2')

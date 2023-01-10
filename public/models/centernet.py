import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.models.backbone import get_backbone
# from public.neck import CenterNetNeck
from public.head import CenterNetHead
from public import neck

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_centernet',
    'resnet34_centernet',
    'resnet50_centernet',
    'resnet101_centernet',
    'resnet152_centernet',
    'swin_t_centernet',
    'swin_s_centernet',
    'swin_b_centernet',
    'swin_l_centernet'
]


# assert input annotations are[x_min,y_min,x_max,y_max]
class CenterNet(nn.Module):
    def __init__(self, backbone_type:str="resnet18", backbone_dict:dict={}, pretrained:bool=False, 
            neck_type:str="CenterNetNeck", neck_dict:dict=None, head_dict:dict=None):
        """
        @description  :
        ---------
        @param  :
            neck_type:
                CenterNetNeck: the original centernet upsample neck
                TTFNeck: ttf neck
            neck_dict: the param in neck
            backbone_dict: the param in backbone
            head_dict: the param in head
            pretrained: use pretrained backbone
        -------
        @Returns  :
        -------
        """

        super(CenterNet, self).__init__()
        self.backbone = get_backbone(backbone_type=backbone_type, pretrained=pretrained, backbone_dict=backbone_dict)
        final_out_size = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048,
            "densecl_resnet50_coco":2048,
            "densecl_resnet50_imagenet":2048,
            "swin_t": 768,
            "swin_s": 768,
            "swin_b": 1024,
            "swin_l": 1536,
        }
        all_inplanes = int(
            final_out_size[backbone_type]/4), int(
                final_out_size[backbone_type]/2), int(
                    final_out_size[backbone_type])
        
        # self.neck = CenterNetNeck(inplanes=C5_inplanes,
        #                         **neck_dict)
        self.neck = neck.__dict__[neck_type](inplanes=all_inplanes,
                                **neck_dict)

        self.head = CenterNetHead(**head_dict)

    def forward(self, inputs):
        backbone_out = self.backbone(inputs)

        neck_out = self.neck(backbone_out)

        del inputs, backbone_out

        heatmap_output, offset_output, wh_output = self.head(neck_out)

        del neck_out

        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # offset_output shape:[3, 2, 160, 160]
        # wh_output shape:[3, 2, 160, 160]

        return heatmap_output, offset_output, wh_output



if __name__ == '__main__':
    neck_dict = dict(out_channels=[256, 128, 64], selayer=True)
    head_dict = dict(num_classes=80,
                 out_channels=[256, 128, 64])
    net = CenterNet(backbone_type="resnet50", neck_type="TTFNeck", neck_dict=neck_dict, head_dict=head_dict)
    image_h, image_w = 512, 512
    heatmap_output, offset_output, wh_output = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", heatmap_output.shape, offset_output.shape, wh_output.shape)

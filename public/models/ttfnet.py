import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# from public.path import pretrained_models_path

from public.models.backbone import get_backbone
from public.head import TTFNetHead
from public import neck

import torch
import torch.nn as nn


# assert input annotations are[x_min,y_min,x_max,y_max]
class TTFNet(nn.Module):
    def __init__(self, backbone_type: str = "resnet18", backbone_dict: dict = {}, pretrained: bool = False,
                 neck_type: str = "TTFNeck", neck_dict: dict = None, head_dict: dict = None):
        """
        @description  :
        ---------
        @param  :
            neck_type:
                TTFNeck: ttf neck
                CenterNetNeck: the original centernet upsample neck
            neck_dict: the param in neck
            backbone_dict: the param in backbone
            head_dict: the param in head
            pretrained: use pretrained backbone
        -------
        @Returns  :
        -------
        """

        super(TTFNet, self).__init__()
        self.backbone = get_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
            backbone_dict=backbone_dict,
        )

        # TODO: a better solution is to cal `self.basebone.get_inplaces()`
        inplanes_dict = {
            'resnet18': (64, 128, 256, 512),
            'resnet34': (64, 128, 256, 512),
            'resnet50': (256, 512, 1024, 2048),
            'resnet101': (256, 512, 1024, 2048),
            'resnet152': (256, 512, 1024, 2048),

            'densecl_resnet50_coco': (256, 512, 1024, 2048),
            'densecl_resnet50_imagenet': (256, 512, 1024, 2048),

            'swin_t': (96, 192, 384, 768),
            'swin_s': (96, 192, 384, 768),
            'swin_b': (128, 256, 512, 1024),
            'swin_l': (192, 384, 768, 1536),

            'rmobilenet': (24, 32, 96, 320),

            # 'repvgg_pruning_a0': (48, 96, 192, 1280),
            # 'repvgg_pruning_a1': (64, 128, 256, 1280),
            # 'repvgg_pruning_b0': (64, 128, 256, 1280),

            # 'rmnet_pruning_18': (64, 128, 256, 512),
            # 'rmnet_pruning_34': (64, 128, 256, 512),

            "mobilenetv2": (24, 32, 96, 320),
            "convnext": (192, 384, 768, 1536)
        }

        # self.neck = CenterNetNeck(inplanes=C5_inplanes, **neck_dict)
        # print(backbone_type, neck_type)
        # print(inplanes_dict[backbone_type])
        self.neck = neck.__dict__[neck_type](inplanes=inplanes_dict[backbone_type], **neck_dict)
        self.head = TTFNetHead(**head_dict)

    def forward(self, inputs):
        # centernet
        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # offset_output shape:[3, 2, 160, 160]
        # wh_output shape:[3, 2, 160, 160]

        # ttfnet
        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # wh_output shape:[3, 4, 160, 160]

        backbone_out = self.backbone(inputs)
        if 0:
            print("backbone")
            return backbone_out

        neck_out = self.neck(backbone_out)
        if 0:
            print("neck")
            return neck_out

        heatmap_output, wh_output = self.head(neck_out)
        # print("head")

        return heatmap_output, wh_output


if __name__ == '__main__':
    neck_dict = dict(out_channels=[256, 128, 64], selayer=True)
    head_dict = dict(num_classes=80,
                     out_channels=[256, 128, 64])
    net = TTFNet(backbone_type="resnet50", neck_type="TTFNeck", neck_dict=neck_dict, head_dict=head_dict)
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

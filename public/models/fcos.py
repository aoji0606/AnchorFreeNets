import os
import sys
import math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.models.backbone import get_backbone
from public import neck
from public.head import FCOSHead

import torch
import torch.nn as nn
import torch.nn.functional as F


# assert input annotations are[x_min,y_min,x_max,y_max]
class FCOS(nn.Module):
    def __init__(self, backbone_type:str="resnet18", neck_type:str="RetinaFPN", neck_dict:dict=None, 
                pretrained:bool=False,head_dict:dict=None, 
                scales:list=[1.0, 1.0, 1.0, 1.0, 1.0], backbone_dict:dict=None):
        """
        @description  :FCOS det
        ---------
        @param  :
            neck_type:
                CenterNetNeck: the original centernet upsample neck
                TTFNeck: ttf neck
            neck_dict: the param in neck
            backbone_dict: the param in backbone
            head_dict: the param in head
            pretrained: use pretrained backbone
            scales: the factor in per stage
        -------
        @Returns  :
        -------
        """

        super(FCOS, self).__init__()

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

        self.neck = neck.__dict__[neck_type](
                            inplanes=all_inplanes,
                            **neck_dict)

        self.head = FCOSHead(**head_dict)

        self.scales = nn.Parameter(
            torch.tensor(np.array(scales), dtype=torch.float32))

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device
        backbone_out = self.backbone(inputs)

        del inputs

        features = self.neck(backbone_out)

        del backbone_out

        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):

            cls_outs, reg_outs, center_outs = self.head(feature)

            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
            reg_outs = reg_outs * torch.exp(scale)
            reg_heads.append(reg_outs)
            # [N,1,H,W] -> [N,H,W,1]
            center_outs = center_outs.permute(0, 2, 3, 1).contiguous()
            center_heads.append(center_outs)

        del features


        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]

        return cls_heads, reg_heads, center_heads



if __name__ == '__main__':
    net = FCOS(backbone_type="resnet50", neck_type="YolofDC5")
    image_h, image_w = 512, 512
    cls_heads, reg_heads, center_heads, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", cls_heads[0].shape, reg_heads[0].shape,
          center_heads[0].shape, batch_positions[0].shape)

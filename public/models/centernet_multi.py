import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.models.backbone import get_backbone
import public.neck as Neck
from public.head import CenterNetHead

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_centernet',
    'resnet34_centernet',
    'resnet50_centernet',
    'resnet101_centernet',
    'resnet152_centernet',
]

model_urls = {
    'resnet18_centernet':
    '{}/detection_models/resnet18dcn_centernet_coco_multi_scale_resize512_mAP0.266.pth'
    .format(pretrained_models_path),
    'resnet34_centernet':
    'empty',
    'resnet50_centernet':
    'empty',
    'resnet101_centernet':
    'empty',
    'resnet152_centernet':
    'empty',
}

# assert input annotations are[x_min,y_min,x_max,y_max]
class CenterNetMul(nn.Module):
    def __init__(self, backbone_type="resnet18", backbone_dict={}, pretrained=False, 
                        head_num=1, neck_dicts=None, head_dicts=None,
                        train_head_pos=1,deploy=False):
        """
        @description  :
        ---------
        @param  :
            head_num: the num of heads
            train_head_pos: choose the head to be trained
            deploy: use all the heads
        -------
        @Returns  :
            if deploy: [[hmap,offset,wh]*head_num]
            else: hmap,offset,wh
        -------
        """
        super(CenterNetMul, self).__init__()
        self.backbone = get_backbone(backbone_type=backbone_type, pretrained=pretrained, backbone_dict=backbone_dict)
        self.deploy = deploy
        self.train_head_pos = train_head_pos
        self.head_num = head_num
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
        
        for i in range(0,head_num):
            cur_neck_dict = neck_dicts[i]
            self.add_module("neck"+str(i+1), 
                            Neck.__dict__[cur_neck_dict["neck_type"]](inplanes=all_inplanes,
                                **(cur_neck_dict["param"])))
        

        for i in range(0,head_num):
            cur_head_dict = head_dicts[i]
            self.add_module("head"+str(i+1), 
                            CenterNetHead(**cur_head_dict))
        
        self.frozen()
    

    def frozen(self):
        if self.train_head_pos != 1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        for i in range(1,self.head_num+1):
            if i != self.train_head_pos:
                neck = getattr(self, "neck"+str(i))
                head = getattr(self, "head"+str(i))
                for param in neck.parameters():
                    param.requires_grad = False
                for param in head.parameters():
                    param.requires_grad = False

    def forward(self, inputs):
        backbone_out = self.backbone(inputs)
        del inputs

        if self.deploy:
            out = []
            for i in range(self.head_num):
                neck = getattr(self, "neck"+str(i+1))
                head = getattr(self, "head"+str(i+1))
                neck_out = neck(backbone_out)
                
                heatmap_output, offset_output, wh_output = head(neck_out)
            
                del neck_out
                out.append([heatmap_output, offset_output, wh_output])
            del backbone_out

            return out
        
        else:
            neck = getattr(self, "neck"+str(self.train_head_pos))
            head = getattr(self, "head"+str(self.train_head_pos))
            neck_out = neck(backbone_out)
            heatmap_output, offset_output, wh_output = head(neck_out)
            del neck_out, backbone_out

            return heatmap_output, offset_output, wh_output
        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # offset_output shape:[3, 2, 160, 160]
        # wh_output shape:[3, 2, 160, 160]


if __name__ == '__main__':
    # neck_dict1 = dict(neck_type = "CenterNetNeck",
    #                 param=dict(num_layers=3,
    #                     out_channels=[256, 128, 64]))
    # neck_dict2 = dict(neck_type = "TTFNeck",
    #                 param=dict(out_channels=[256, 128, 64],selayer=False))
    # neck_dicts = [neck_dict1,neck_dict2]
    # head_dict1 = dict(num_classes=80,
    #              out_channels=[256, 128, 64])
    # head_dict2 = dict(num_classes=20,
    #              out_channels=[256, 128, 64])
    # head_dicts = [head_dict1, head_dict2]
    # net = CenterNetMul(backbone_type="swin_t",backbone_dict=dict(out_indices=(3,)), 
    #                     neck_dicts=neck_dicts, head_dicts=head_dicts,
    #                     head_num=2,
    #                     train_head_pos=1,
    #                     deploy=False)
    net = CenterNetMul(backbone_type=Config.backbone_type, backbone_dict=Config.backbone_dict, 
                        neck_dicts=Config.neck_dicts, head_dicts=Config.head_dicts,
                        head_num=2,
                        train_head_pos=1,
                        deploy=False)
    image_h, image_w = 512, 512
    # heatmap_output, offset_output, wh_output = net(
    #     torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))

    for i in range(2):
        heatmap_output, offset_output, wh_output = out[i]
        print("1111", heatmap_output.shape, offset_output.shape, wh_output.shape)


import os
import sys

# import numpy as np

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.models.backbone import get_backbone
from public.head.TTFNetHeadPruning import TTFNetHeadPruning
from public import neck


# assert input annotations are[x_min,y_min,x_max,y_max]
class TTFNetPruning(nn.Module):
    def __init__(self, backbone_type:str="resnet18", backbone_dict:dict={}, pretrained:bool=False, 
            neck_type:str="TTFNeckPruning", neck_dict:dict=None, head_dict:dict=None):
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

        super(TTFNetPruning, self).__init__()
        self.backbone = get_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
            backbone_dict=backbone_dict,
        )

        # TODO: a better solution is to cal `self.basebone.get_inplaces()`
        inplanes_dict = {
            # 'resnet18': (64, 128, 256, 512),
            # 'resnet34': (64, 128, 256, 512),
            # 'resnet50': (256, 512, 1024, 2048),
            # 'resnet101': (256, 512, 1024, 2048),
            # 'resnet152': (256, 512, 1024, 2048),
            
            # 'densecl_resnet50_coco': (256, 512, 1024, 2048),
            # 'densecl_resnet50_imagenet': (256, 512, 1024, 2048),

            # 'swin_t': (96, 192, 384, 768),
            # 'swin_s': (96, 192, 384, 768),
            # 'swin_b': (128, 256, 512, 1024),
            # 'swin_l': (192, 384, 768, 1536),
            
            'repvgg_pruning_a0': (48, 96, 192, 1280),
            'repvgg_pruning_a1': (64, 128, 256, 1280),
            'repvgg_pruning_b0': (64, 128, 256, 1280),

            'rmnet_pruning_18': (64, 128, 256, 512),
            'rmnet_pruning_34': (64, 128, 256, 512),

            'rmobilenet_pruning': (24, 32, 96, 320),
        }

        # self.neck = CenterNetNeck(inplanes=C5_inplanes, **neck_dict)
        self.neck = neck.__dict__[neck_type](inplanes=inplanes_dict[backbone_type], **neck_dict)
        self.head = TTFNetHeadPruning(**head_dict)

    def forward(self, inputs):
        backbone_out = self.backbone(inputs)
        neck_out = self.neck(backbone_out)
        heatmap_output, wh_output = self.head(neck_out)

        # centernet
        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # offset_output shape:[3, 2, 160, 160]
        # wh_output shape:[3, 2, 160, 160]

        # ttfnet
        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # wh_output shape:[3, 4, 160, 160]
        return heatmap_output, wh_output
    
    def deploy_model(self):
        # model = copy.deepcopy(self)
        for module in self.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        # return model
    
    def update_mask(self, sr, threshold):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1 and m.weight.requires_grad:
                m.weight.grad.data.add_(sr * torch.sign(m.weight.data))
                m1 = m.weight.data.abs() > threshold
                m.weight.grad.data *= m1
                m.weight.data *= m1
    
    def fix_mask(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1:
                m.weight.requires_grad = False
    
    def prune_by_thresh(self, thresh=0):
        prune_cnt, total_cnt = 0, 0

        # prune backbone
        in_mask = torch.ones(3) > 0
        ret_masks, _prune_cnt, _total_cnt = self.backbone.prune_by_mask_and_thresh(in_mask, thresh=thresh)
        prune_cnt += _prune_cnt
        total_cnt += _total_cnt
        print(f'=> backbone pruning ratio: {1 - _prune_cnt / _total_cnt}')

        # prune neck
        in_masks = ret_masks
        ret_masks, _prune_cnt, _total_cnt = self.neck.prune_by_mask_and_thresh(in_masks, thresh=thresh)
        prune_cnt += _prune_cnt
        total_cnt += _total_cnt
        print(f'=> neck pruning ratio: {1 - _prune_cnt / _total_cnt}')

        # prune head
        in_masks = ret_masks
        ret_masks, _prune_cnt, _total_cnt = self.head.prune_by_mask_and_thresh(in_masks, thresh=thresh)
        prune_cnt += _prune_cnt
        total_cnt += _total_cnt
        print(f'=> head pruning ratio: {1 - _prune_cnt / _total_cnt}')

        print(f'=> total pruning ratio: {1 - prune_cnt / total_cnt}')

        return [], prune_cnt, total_cnt
    
    def prune(self, ratio=None, merge_mask=True):
        thresh = 0  # if ratio is none, use static thresh, thresh = 0
        if ratio is not None:
            ##### calculate total mask count #####
            total = 0
            for _, m in self.named_modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1:  # and m.weight.requires_grad:
                    total += m.weight.data.shape[0]
            
            ##### find thresh by ratio #####
            _mask = torch.zeros(total); index = 0
            for _, m in self.named_modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1:  # and m.weight.requires_grad:
                    size = m.weight.data.shape[0]
                    _mask[index:(index + size)] = m.weight.data.reshape(-1).abs().clone()
                    index += size
            
            y, _ = torch.sort(_mask)
            thre_index = int(total * ratio)
            thresh = y[thre_index]
        
        print(f'=> pruning thresh: {thresh}')

        self.backbone.deploy_model()

        self.prune_by_thresh(thresh)

        if merge_mask:
            self.backbone.merge_masks()
            self.neck.merge_masks()
            self.head.merge_masks()

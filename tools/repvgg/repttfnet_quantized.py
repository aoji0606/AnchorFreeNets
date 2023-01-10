import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

from typing import List, Tuple


class RepTTFNetNeckQuant(nn.Module):

    def __init__(self, ori_model):
        super(RepTTFNetNeckQuant, self).__init__()
        self.up5t4 = ori_model.up5t4
        self.up4t3 = ori_model.up4t3
        self.up3t2 = ori_model.up3t2
        
        self.shortcut4 = ori_model.shortcut4
        self.shortcut3 = ori_model.shortcut3
        self.shortcut2 = ori_model.shortcut2

        self.functional = torch.nn.quantized.FloatFunctional()
    
    def forward(self, x:List[torch.Tensor]) -> List[torch.Tensor]:
        C2, C3, C4, C5 = x
        C4_up = self.functional.add(self.up5t4(C5   ), self.shortcut4(C4))
        C3_up = self.functional.add(self.up4t3(C4_up), self.shortcut3(C3))
        C2_up = self.functional.add(self.up3t2(C3_up), self.shortcut2(C2))
        return [C2_up]


class RepTTFNetHeadQuant(nn.Module):

    def __init__(self, ori_model):
        super(RepTTFNetHeadQuant, self).__init__()
        self.heatmap_head = ori_model.heatmap_head
        self.wh_head = ori_model.wh_head
        self.wh_offset_base = float(ori_model.wh_offset_base) # mul_scalar function need a float input

        self.relu = nn.ReLU()
        self.functional = torch.nn.quantized.FloatFunctional()
    
    def forward(self, x:List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        heatmap_output = self.heatmap_head(x[-1])
        wh_output = self.functional.mul_scalar(
            self.relu(self.functional.add_scalar(self.wh_head(x[-1]), 0.1)),
            self.wh_offset_base
        )  # scale wh
        return heatmap_output, wh_output


class RepTTFNetWholeQuant(nn.Module):

    def __init__(self, repttfnet_model, quantlayers='all'):
        super(RepTTFNetWholeQuant, self).__init__()
        # assert quantlayers in ['all', 'exclud_first_and_linear', 'exclud_first_and_last']
        # self.quantlayers = quantlayers
        self.quant = QuantStub()
        self.backbone = repttfnet_model.backbone
        self.neck = RepTTFNetNeckQuant(repttfnet_model.neck)
        self.head = RepTTFNetHeadQuant(repttfnet_model.head)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        heatmap_output, wh_output = self.head(neck_out)
        heatmap_output = self.dequant(heatmap_output)
        wh_output = self.dequant(wh_output)
        return heatmap_output, wh_output

    #   From https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential) and hasattr(m, 'conv'):
                # Note that we moved ReLU from "block.nonlinearity" into "rbr_reparam" (nn.Sequential).
                # This makes it more convenient to fuse operators using off-the-shelf APIs.
                torch.quantization.fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)

    def _get_qconfig(self):
        return torch.quantization.get_default_qat_qconfig('fbgemm')

    def prepare_quant(self):
        #   From https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
        self.fuse_model()
        qconfig = self._get_qconfig()
        self.qconfig = qconfig
        torch.quantization.prepare_qat(self, inplace=True)

    def freeze_quant_bn(self):
        self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
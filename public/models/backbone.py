import os
import sys

import torch
import torch.nn as nn
import torchvision

import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public import backbone
from public.backbone.rmnet_pruning import ResBlock


def _prune_dw_conv_bn(conv, bn, mask):
    temp_conv = nn.Conv2d(
        int(mask.sum()),
        int(mask.sum()),
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=int(mask.sum()),
        bias=False
    )
    temp_conv.weight.data = conv.weight.data[mask]

    temp_bn = None
    if bn is not None:
        temp_bn = nn.BatchNorm2d(int(mask.sum()))
        temp_bn.weight.data = bn.weight.data[mask]
        temp_bn.bias.data = bn.bias.data[mask]
        temp_bn.running_mean = bn.running_mean[mask]
        temp_bn.running_var = bn.running_var[mask]

    return temp_conv, temp_bn


def _prune_conv_bn_mask(conv=None, bn=None, mask=None, in_mask=None, out_mask=None):
    temp_conv = None
    if conv is not None:
        temp_conv = nn.Conv2d(
            int(in_mask.sum()),
            int(out_mask.sum()),
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=False
        )
        temp_conv.weight.data = conv.weight.data[out_mask][:, in_mask]

    temp_bn = None
    if bn is not None:
        temp_bn = nn.BatchNorm2d(int(out_mask.sum()))
        temp_bn.weight.data = bn.weight.data[out_mask]
        temp_bn.bias.data = bn.bias.data[out_mask]
        temp_bn.running_mean = bn.running_mean[out_mask]
        temp_bn.running_var = bn.running_var[out_mask]

    temp_mask = None
    if mask is not None:
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


class Darknet19Backbone(nn.Module):
    def __init__(self):
        super(Darknet19Backbone, self).__init__()
        self.model = backbone.__dict__['darknet19'](**{"pretrained": True})
        del self.model.avgpool
        del self.model.layer7

    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.maxpool1(x)
        x = self.model.layer2(x)
        C3 = self.model.layer3(x)
        C4 = self.model.layer4(C3)
        C5 = self.model.layer5(C4)
        C5 = self.model.layer6(C5)

        return [C2, C3, C4, C5]


class Darknet53Backbone(nn.Module):
    def __init__(self):
        super(Darknet53Backbone, self).__init__()
        self.model = backbone.__dict__['darknet53'](**{"pretrained": True})
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.block1(x)
        x = self.model.conv3(x)
        x = self.model.block2(x)
        x = self.model.conv4(x)
        C3 = self.model.block3(x)
        C4 = self.model.conv5(C3)
        C4 = self.model.block4(C4)
        C5 = self.model.conv6(C4)
        C5 = self.model.block5(C5)

        return [C2, C3, C4, C5]


class EfficientNetBackbone(nn.Module):
    def __init__(self, efficientnet_type="efficientnet_b0"):
        super(EfficientNetBackbone, self).__init__()
        self.model = backbone.__dict__[efficientnet_type](**{"pretrained": True})
        del self.model.dropout
        del self.model.fc
        del self.model.avgpool
        del self.model.conv_head

    def forward(self, x):
        x = self.model.stem(x)

        feature_maps = []
        last_x = None
        for index, block in enumerate(self.model.blocks):
            x = block(x)
            if block.stride == 2:
                feature_maps.append(last_x)
            elif index == len(self.model.blocks) - 1:
                feature_maps.append(x)
            last_x = x

        return feature_maps[2:]


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type='resnet50', pretrained=False):
        super(ResNetBackbone, self).__init__()
        model = backbone.__dict__[resnet_type](**{'pretrained': pretrained})

        self.model = nn.ModuleDict({
            'conv1': model.conv1,
            'bn1': model.bn1,
            'relu': model.relu,
            'maxpool': model.maxpool,
            'layer1': model.layer1,
            'layer2': model.layer2,
            'layer3': model.layer3,
            'layer4': model.layer4,
        })

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        C2 = self.model.layer1(x)
        C3 = self.model.layer2(C2)
        C4 = self.model.layer3(C3)
        C5 = self.model.layer4(C4)

        return [C2, C3, C4, C5]


class MobileNetV2Backbone(nn.Module):
    def __init__(self, mobilenet_type='mobilenetv2', pretrained=False):
        super(MobileNetV2Backbone, self).__init__()
        # model = backbone.__dict__[mobilenet_type](**{'pretrained': pretrained})
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)

        self.model = nn.ModuleDict({})
        for i in range(18):
            self.model["feature%d" % i] = model.features[i]

    def forward(self, x):
        features = []

        for i, (k, v) in enumerate(self.model.items()):
            x = v(x)
            if i in [3, 6, 13, 17]:  # 320
                features.append(x)

        return features


class MobileOneBackbone(nn.Module):
    def __init__(self, type='', pretrained=False):
        super(MobileOneBackbone, self).__init__()
        model = backbone.mobileone(variant="s1", inference_mode=False)
        model.load_state_dict(torch.load("../public/models/mobileone_s1_unfused.pth.tar", "cpu"))

        self.model = nn.ModuleDict({
            'stage0': model.stage0,
            'stage1': model.stage1,
            'stage2': model.stage2,
            'stage3': model.stage3,
            'stage4': model.stage4,
        })

    def forward(self, x):
        x0 = self.model.stage0(x)
        x1 = self.model.stage1(x0)
        x2 = self.model.stage2(x1)
        x3 = self.model.stage3(x2)
        x4 = self.model.stage4(x3)

        return [x1, x2, x3, x4]


class ConvNextBackbone(nn.Module):
    def __init__(self, type='', pretrained=False):
        super(ConvNextBackbone, self).__init__()
        model = torchvision.models.convnext_large(pretrained=pretrained)

        self.model = nn.ModuleDict({})
        for i in range(8):
            self.model["feature%d" % i] = model.features[i]

    def forward(self, x):
        features = []

        for i, (k, v) in enumerate(self.model.items()):
            x = v(x)
            if i in [1, 3, 5, 7]:
                features.append(x)

        return features


class VovNetBackbone(nn.Module):
    def __init__(self, vovnet_type='VoVNet39_se'):
        super(VovNetBackbone, self).__init__()
        self.model = backbone.__dict__[vovnet_type](**{"pretrained": True})
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.stem(x)

        features = []
        for stage in self.model.stages:
            x = stage(x)
            features.append(x)

        del x

        return features[1:]


class SwinTransformer(nn.Module):
    def __init__(self, swin_type="swin_t", pretrained=False, backbone_dict={}):
        super(SwinTransformer, self).__init__()
        self.model = backbone.__dict__[swin_type](pretrained=pretrained, backbone_dict=backbone_dict)

    def forward(self, x):
        out = self.model(x)
        del x
        return out


class RepVGGPruning(nn.Module):
    def __init__(self, repvgg_type='repvgg_pruning_a0', pretrained=False):
        super(RepVGGPruning, self).__init__()

        model = backbone.__dict__[repvgg_type](pretrained=pretrained)
        self.model = nn.ModuleDict({
            'stage0': model.stage0,
            'stage1': model.stage1,
            'stage2': model.stage2,
            'stage3': model.stage3,
            'stage4': model.stage4
        })

    def forward(self, x):
        out = self.model.stage0(x)
        C2 = self.model.stage1(out)
        C3 = self.model.stage2(C2)
        C4 = self.model.stage3(C3)
        C5 = self.model.stage4(C4)
        return [C2, C3, C4, C5]

    def deploy_model(self):
        # model = copy.deepcopy(self)
        for module in self.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        # return model

    def update_mask(self, sr, threshold):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1 and m.weight.requires_grad:
                m.weight.grad.data.add_(sr * torch.sign(m.weight.data))
                m1 = m.weight.data > threshold
                m.weight.grad.data *= m1
                m.weight.data *= m1

    def fix_mask(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1:
                m.weight.requires_grad = False

    def get_custom_l2(self, weight_decay):
        loss = 0
        for module in self.modules():
            if hasattr(module, 'get_custom_L2'):
                loss += weight_decay * 0.5 * module.get_custom_L2()
        return loss

    def fix_layer_last_mask(self):
        self.model.stage1[-1].mask.weight.requires_grad = False
        self.model.stage2[-1].mask.weight.requires_grad = False
        self.model.stage3[-1].mask.weight.requires_grad = False
        self.model.stage4[-1].mask.weight.requires_grad = False

    def prune_by_thresh(self, thresh, fixed_layers=[]):
        from public.backbone.repvgg_pruning import RepVGGBlock, conv_bn

        in_mask = torch.ones(3) > 0
        prune_cnt = 0;
        total_cnt = 0
        for n, m in self.model.named_modules():
            if isinstance(m, RepVGGBlock) and n in fixed_layers:
                mask = m.rbr_reparam.bn.weight.data.abs().reshape(-1) > -1  # all True
                in_mask = mask  # next input mask

            elif isinstance(m, RepVGGBlock) and n not in fixed_layers:
                mask = m.rbr_reparam.bn.weight.data.abs().reshape(-1) > thresh

                prune_cnt += int(mask.sum())
                total_cnt += int(len(mask))

                # prune model
                temp_param = conv_bn(
                    int(in_mask.sum()),
                    int(mask.sum()),
                    m.rbr_reparam.conv.kernel_size,
                    m.rbr_reparam.conv.stride,
                    m.rbr_reparam.conv.padding,
                    m.rbr_reparam.conv.dilation,
                    m.rbr_reparam.conv.groups,
                    bias=False
                )
                temp_param.conv.weight.data = m.rbr_reparam.conv.weight.data[mask][:, in_mask]
                temp_param.bn.weight.data = m.rbr_reparam.bn.weight.data[mask]
                temp_param.bn.bias.data = m.rbr_reparam.bn.bias.data[mask]
                temp_param.bn.running_mean = m.rbr_reparam.bn.running_mean[mask]
                temp_param.bn.running_var = m.rbr_reparam.bn.running_var[mask]
                m.rbr_reparam = temp_param

                in_mask = mask  # next input mask

        print(f'=> pruning ratio: {1 - prune_cnt / total_cnt}')

    def prune(self, ratio=None):
        fixed_layers = [
            f'model.stage1.{len(self.model.stage1) - 1}',
            f'model.stage2.{len(self.model.stage2) - 1}',
            f'model.stage3.{len(self.model.stage3) - 1}',
            f'model.stage4.{len(self.model.stage4) - 1}',
        ]

        from public.backbone.repvgg_pruning import RepVGGBlock

        self.deploy_model()

        thresh = 0  # if ratio is none, use static thresh, thresh = 0
        if ratio is not None:
            ##### calculate total mask count #####
            total = 0
            for n, m in self.model.named_modules():
                if isinstance(m, RepVGGBlock) and n not in fixed_layers:
                    total += m.rbr_reparam.bn.weight.data.shape[0]

            ##### find thresh by ratio #####
            _mask = torch.zeros(total);
            index = 0
            for n, m in self.model.named_modules():
                if isinstance(m, RepVGGBlock) and n not in fixed_layers:
                    size = m.rbr_reparam.bn.weight.data.shape[0]
                    _mask[index:(index + size)] = m.rbr_reparam.bn.weight.data.abs().clone()
                    index += size
            y, _ = torch.sort(_mask)
            thre_index = int(total * ratio)
            thresh = y[thre_index]
        print(f'=> pruning thresh: {thresh}')

        self.prune_by_thresh(thresh, fixed_layers)


class RMNetPruning(nn.Module):
    def __init__(self, rmnet_type='rmnet_pruning_18', pretrained=False):
        super(RMNetPruning, self).__init__()

        model = backbone.__dict__[rmnet_type](pretrained=pretrained)
        self.model = nn.ModuleDict({
            'conv1': model.conv1,
            'mask1': model.mask1,
            'bn1': model.bn1,
            'relu': model.relu,
            'maxpool': model.maxpool,
            'layer1': model.layer1,
            'layer2': model.layer2,
            'layer3': model.layer3,
            'layer4': model.layer4,
        })

    def forward(self, x):
        x = self.model.bn1(self.model.conv1(x))
        if hasattr(self.model, 'mask1'):
            x = self.model.mask1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        C2 = self.model.layer1(x)
        C3 = self.model.layer2(C2)
        C4 = self.model.layer3(C3)
        C5 = self.model.layer4(C4)

        return [C2, C3, C4, C5]

    def deploy_model(self, keep_mask=True):
        if not keep_mask:
            # merge mask1
            self.model.bn1.weight.data *= self.model.mask1.weight.data.reshape(-1)
            self.model.bn1.bias.data *= self.model.mask1.weight.data.reshape(-1)
            self.model.pop('mask1')

        # merge layers
        for module in self.model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy(keep_mask=keep_mask)

    def merge_masks(self):
        # first conv
        self.model.bn1.weight.data *= self.model.mask1.weight.data.reshape(-1)
        self.model.bn1.bias.data *= self.model.mask1.weight.data.reshape(-1)
        self.model.__delattr__('mask1')
        # other layers
        for _, m in self.model.named_modules():
            if isinstance(m, ResBlock):
                m.reparam.bn1.weight.data *= m.reparam.mask1.weight.data.reshape(-1)
                m.reparam.bn1.bias.data *= m.reparam.mask1.weight.data.reshape(-1)
                m.reparam.bn2.weight.data *= m.reparam.mask2.weight.data.reshape(-1)
                m.reparam.bn2.bias.data *= m.reparam.mask2.weight.data.reshape(-1)
                m.reparam = nn.Sequential(*[
                    e for i, e in enumerate(m.reparam.children()) if i in [0, 1, 3, 4, 5, 7]
                ])

    def prune_by_mask_and_thresh(self, in_mask, thresh=0):
        # out mask
        ret_masks = [
            self.model.layer1[-1].reparam.mask2.weight.data.abs().reshape(-1) > thresh,
            self.model.layer2[-1].reparam.mask2.weight.data.abs().reshape(-1) > thresh,
            self.model.layer3[-1].reparam.mask2.weight.data.abs().reshape(-1) > thresh,
            self.model.layer4[-1].reparam.mask2.weight.data.abs().reshape(-1) > thresh,
        ]

        prune_cnt = 0;
        total_cnt = 0

        # first conv
        mask = self.model.mask1.weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        temp_conv, temp_bn, temp_mask = _prune_conv_bn_mask(self.model.conv1, self.model.bn1, self.model.mask1, in_mask,
                                                            mask)
        self.model.conv1 = temp_conv
        self.model.bn1 = temp_bn
        self.model.mask1 = temp_mask
        in_mask = mask

        # other layers
        for _, m in self.model.named_modules():
            if isinstance(m, ResBlock):
                # mask1
                mask1 = m.reparam.mask1.weight.data.abs().reshape(-1) > thresh
                prune_cnt += int(mask1.sum())
                total_cnt += int(len(mask1))

                temp_conv, temp_bn, temp_mask = _prune_conv_bn_mask(m.reparam.conv1, m.reparam.bn1, m.reparam.mask1,
                                                                    in_mask, mask1)
                m.reparam.conv1 = temp_conv
                m.reparam.bn1 = temp_bn
                m.reparam.mask1 = temp_mask
                in_mask = mask1

                # mask2
                mask2 = m.reparam.mask2.weight.data.abs().reshape(-1) > thresh
                prune_cnt += int(mask2.sum())
                total_cnt += int(len(mask2))

                temp_conv, temp_bn, temp_mask = _prune_conv_bn_mask(m.reparam.conv2, m.reparam.bn2, m.reparam.mask2,
                                                                    in_mask, mask2)
                m.reparam.conv2 = temp_conv
                m.reparam.bn2 = temp_bn
                m.reparam.mask2 = temp_mask
                in_mask = mask2

        return ret_masks, prune_cnt, total_cnt


class RMobilenet(nn.Module):
    def __init__(self, mobilenet_type='rmobilenet', pretrained=False):
        super(RMobilenet, self).__init__()
        model = backbone.__dict__[mobilenet_type](**{'pretrained': pretrained})

        self.stem = list()
        self.layer1 = list()
        self.layer2 = list()
        self.layer3 = list()
        self.layer4 = list()

        for i, m in enumerate(model.features):
            if i in [0, 1]:
                self.stem.append(m)
            elif i in [2, 3]:
                self.layer1.append(m)
            elif i in [4, 5, 6]:
                self.layer2.append(m)
            elif i in [7, 8, 9, 10, 11, 12, 13]:
                self.layer3.append(m)
            elif i in [14, 15, 16, 17]:
                self.layer4.append(m)

        self.stem = nn.Sequential(*self.stem)
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)
        self.layer4 = nn.Sequential(*self.layer4)

        # out: [24, 32, 96, 320]

    def forward(self, x):
        x = self.stem(x)
        C2 = self.layer1(x)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return [C2, C3, C4, C5]

    def deploy_model(self):
        from public.backbone.rmobilenet import InvertedResidual

        for m in self.features:
            if isinstance(m, InvertedResidual):
                m.deploy()

        # TODO: merge conv,bn,conv,bn style
        return self


class RMobilenetPruning(nn.Module):
    def __init__(self, mobilenet_type='rmobilenet_pruning', pretrained=False):
        super(RMobilenetPruning, self).__init__()
        model = backbone.__dict__[mobilenet_type](**{'pretrained': pretrained})

        self.stem = list()
        self.layer1 = list()
        self.layer2 = list()
        self.layer3 = list()
        self.layer4 = list()

        for i, m in enumerate(model.features):
            if i in [0, 1]:
                self.stem.append(m)
            elif i in [2, 3]:
                self.layer1.append(m)
            elif i in [4, 5, 6]:
                self.layer2.append(m)
            elif i in [7, 8, 9, 10, 11, 12, 13]:
                self.layer3.append(m)
            elif i in [14, 15, 16, 17]:
                self.layer4.append(m)

        self.stem = nn.Sequential(*self.stem)
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)
        self.layer4 = nn.Sequential(*self.layer4)
        # out: [24, 32, 96, 320]

    def forward(self, x):
        x = self.stem(x)
        C2 = self.layer1(x)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return [C2, C3, C4, C5]

    def deploy_model(self, keep_mask=True):
        from public.backbone.rmobilenet_pruning import InvertedResidualPruning

        for m in self.modules():
            if isinstance(m, InvertedResidualPruning):
                m.deploy(keep_mask=keep_mask)

        # TODO: merge conv,bn,conv,bn style
        return self

    def merge_masks(self):
        # TODO
        from public.backbone.rmobilenet_pruning import InvertedResidualPruning

        # stem
        self.stem[1][1].weight.data *= self.stem[1][2].weight.data.reshape(-1)
        self.stem[1][1].bias.data *= self.stem[1][2].weight.data.reshape(-1)
        self.stem[1][5].weight.data *= self.stem[1][6].weight.data.reshape(-1)
        self.stem[1][5].bias.data *= self.stem[1][6].weight.data.reshape(-1)
        self.stem[1] = nn.Sequential(*[
            self.stem[1][0], self.stem[1][1], self.stem[1][3],
            self.stem[1][4], self.stem[1][5]
        ])

        # other layers
        for _, m in self.named_modules():
            if isinstance(m, InvertedResidualPruning):
                m.dw_conv.bn.weight.data *= m.pw_dw_mask.weight.data.reshape(-1)
                m.dw_conv.bn.bias.data *= m.pw_dw_mask.weight.data.reshape(-1)

                m.pw_linear.bn.weight.data *= m.out_mask.weight.data.reshape(-1)
                m.pw_linear.bn.bias.data *= m.out_mask.weight.data.reshape(-1)

                # m.conv[9].weight.data *= m.conv[10].weight.data.reshape(-1)
                # m.conv[9].bias.data *= m.conv[10].weight.data.reshape(-1)
                # m.conv = nn.Sequential(*[
                #     e for i, e in enumerate(m.conv.children()) if i in [0, 1, 3, 4, 5, 7, 8, 9]
                # ])
                m.__delattr__('pw_dw_mask')
                m.__delattr__('out_mask')

    def prune_by_mask_and_thresh(self, in_mask, thresh=0):
        from public.backbone.rmobilenet_pruning import InvertedResidualPruning

        # out mask
        ret_masks = [
            self.layer1[-1].out_mask.weight.data.abs().reshape(-1) > thresh,
            self.layer2[-1].out_mask.weight.data.abs().reshape(-1) > thresh,
            self.layer3[-1].out_mask.weight.data.abs().reshape(-1) > thresh,
            self.layer4[-1].out_mask.weight.data.abs().reshape(-1) > thresh,
        ]

        prune_cnt = 0;
        total_cnt = 0

        # stem conv
        mask = self.stem[1][2].weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))
        temp_conv, temp_bn, temp_mask = _prune_conv_bn_mask(self.stem[0][0], self.stem[0][1], self.stem[1][2], in_mask,
                                                            mask)
        self.stem[0][0] = temp_conv
        self.stem[0][1] = temp_bn
        self.stem[1][2] = temp_mask
        temp_conv, temp_bn = _prune_dw_conv_bn(self.stem[1][0], self.stem[1][1], mask)
        self.stem[1][0] = temp_conv
        self.stem[1][1] = temp_bn
        in_mask = mask

        mask = self.stem[1][6].weight.data.abs().reshape(-1) > thresh
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))
        temp_conv, temp_bn, temp_mask = _prune_conv_bn_mask(self.stem[1][4], self.stem[1][5], self.stem[1][6], in_mask,
                                                            mask)
        self.stem[1][4] = temp_conv
        self.stem[1][5] = temp_bn
        self.stem[1][6] = temp_mask
        in_mask = mask

        # other layers
        for _, m in self.named_modules():
            if isinstance(m, InvertedResidualPruning):
                # mask1, mask2
                mask1 = m.pw_dw_mask.weight.data.abs().reshape(-1) > thresh
                prune_cnt += int(mask1.sum())
                total_cnt += int(len(mask1))

                m.pw_conv.conv, m.pw_conv.bn, m.pw_dw_mask = _prune_conv_bn_mask(m.pw_conv.conv, m.pw_conv.bn,
                                                                                 m.pw_dw_mask, in_mask, mask1)
                m.dw_conv.conv, m.dw_conv.bn = _prune_dw_conv_bn(m.dw_conv.conv, m.dw_conv.bn, mask1)

                if isinstance(m.pw_conv.act, nn.PReLU):
                    temp_act = nn.PReLU(int(mask1.sum()))
                    temp_act.weight.data = m.pw_conv.act.weight.data[mask1]
                    m.pw_conv.act = temp_act

                if isinstance(m.dw_conv.act, nn.PReLU):
                    temp_act = nn.PReLU(int(mask1.sum()))
                    temp_act.weight.data = m.dw_conv.act.weight.data[mask1]
                    m.dw_conv.act = temp_act

                in_mask = mask1

                # mask3
                mask2 = m.out_mask.weight.data.abs().reshape(-1) > thresh
                prune_cnt += int(mask2.sum())
                total_cnt += int(len(mask2))

                m.pw_linear.conv, m.pw_linear.bn, m.out_mask = _prune_conv_bn_mask(m.pw_linear.conv, m.pw_linear.bn,
                                                                                   m.out_mask, in_mask, mask2)
                in_mask = mask2

        return ret_masks, prune_cnt, total_cnt


def get_backbone(backbone_type, pretrained, backbone_dict={}):
    if 'swin' in backbone_type:
        return SwinTransformer(backbone_type, pretrained, backbone_dict=backbone_dict)
    elif 'resnet' in backbone_type:
        return ResNetBackbone(backbone_type, pretrained)
    elif 'repvgg_pruning' in backbone_type:
        return RepVGGPruning(backbone_type, pretrained)
    elif 'rmnet_pruning' in backbone_type:
        return RMNetPruning(backbone_type, pretrained)
    elif 'rmobilenet' == backbone_type:
        return RMobilenet(backbone_type, pretrained)
    elif 'rmobilenet_pruning' == backbone_type:
        return RMobilenetPruning(backbone_type, pretrained)
    elif "mobilenetv2" in backbone_type:
        return MobileNetV2Backbone(backbone_type, pretrained)
    elif "convnext" in backbone_type:
        return ConvNextBackbone(backbone_type, pretrained)
    elif "mobileone" in backbone_type:
        return MobileOneBackbone(backbone_type, pretrained)


if __name__ == '__main__':
    # net = ResNetBackbone(resnet_type="resnet50")
    # images = torch.randn(8, 3, 640, 640)
    # [C3, C4, C5] = net(images)
    # print("1111", C3.shape, C4.shape, C5.shape)
    # net = EfficientNetBackbone(efficientnet_type="efficientnet_b0")
    # images = torch.randn(8, 3, 640, 640)
    # [C3, C4, C5] = net(images)
    # print("1111", C3.shape, C4.shape, C5.shape)
    net1 = SwinTransformer()

    print(net1.modules)
    images = torch.randn(8, 3, 416, 416)
    [C2, C3, C4, C5] = net1(images)
    # torch.save(net1, "/home/jovyan/data-vol-1/zhangze/notebook/swin_transformer.pth")
    # [C3, C4, C5] = net1(images)
    print("1111", C3.shape, C4.shape, C5.shape)

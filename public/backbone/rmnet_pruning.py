import os, sys
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path


__all__ = ['rmnet_pruning_18', 'rmnet_pruning_34']


model_urls = {
    'rmnet_pruning_18': '{}/resnet/resnet18-epoch100-acc70.316.pth'.format(pretrained_models_path),
    'rmnet_pruning_34': '{}/resnet/resnet34-epoch100-acc73.736.pth'.format(pretrained_models_path),
}


class ResBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()

        assert mid_planes > in_planes

        self.in_planes = in_planes
        self.mid_planes = mid_planes - out_planes + in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, self.mid_planes - in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes - in_planes)
        self.mask1 = nn.Conv2d(self.mid_planes - in_planes, self.mid_planes - in_planes, 1, groups=self.mid_planes - in_planes,bias=False)
        
        self.conv2 = nn.Conv2d(self.mid_planes - in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.mask2 = nn.Conv2d(out_planes, out_planes, 1, groups=out_planes, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if self.in_planes != self.out_planes or self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
            
        self.mask_res = nn.Sequential(*[
            nn.Conv2d(self.in_planes,self.in_planes,1,groups=self.in_planes,bias=False),
            nn.ReLU(inplace=True)
        ])
        self.running1 = nn.BatchNorm2d(in_planes,affine=False)
        self.running2 = nn.BatchNorm2d(out_planes,affine=False)
        nn.init.ones_(self.mask1.weight)
        nn.init.ones_(self.mask2.weight)
        nn.init.ones_(self.mask_res[0].weight)

        self.reparam = None
        
    def forward(self, x):
        # deploy path
        if self.reparam is not None:
            return self.reparam(x)

        # train path
        if self.in_planes == self.out_planes and self.stride == 1:
            self.running1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mask1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(self.mask_res(x))

        self.running2(out)
        out = self.mask2(out)
        out = self.relu(out)
        
        return out
    
    def switch_to_deploy(self, keep_mask=False):
        # idconv1, idbn1, mask1
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.mid_planes).eval()
        
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes] = bn_var_sqrt
        idbn1.bias.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_var.data[:self.in_planes] = self.running1.running_var
        
        idconv1.weight.data[self.in_planes:] = self.conv1.weight.data
        idbn1.weight.data[self.in_planes:] = self.bn1.weight.data
        idbn1.bias.data[self.in_planes:] = self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:] = self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:] = self.bn1.running_var
        
        mask1 = nn.Conv2d(self.mid_planes, self.mid_planes, 1, groups=self.mid_planes, bias=False)
        mask1.weight.data[:self.in_planes] = self.mask_res[0].weight.data * (self.mask_res[0].weight.data > 0)
        mask1.weight.data[self.in_planes:] = self.mask1.weight.data

        if not keep_mask:
            idbn1.weight.data *= mask1.weight.data.reshape(-1)
            idbn1.bias.data *= mask1.weight.data.reshape(-1)

        # idconv2, idbn2, mask2
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.out_planes).eval()
        downsample_bias = 0
        if self.in_planes == self.out_planes:
            nn.init.dirac_(idconv2.weight.data[:, :self.in_planes])
        else:
            idconv2.weight.data[:, :self.in_planes], downsample_bias = self.fuse(
                F.pad(self.downsample[0].weight.data, [1, 1, 1, 1]),
                self.downsample[1].running_mean,
                self.downsample[1].running_var,
                self.downsample[1].weight,
                self.downsample[1].bias,
                self.downsample[1].eps
            )

        idconv2.weight.data[:,self.in_planes:], bias = self.fuse(
            self.conv2.weight,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.bn2.weight,
            self.bn2.bias,
            self.bn2.eps
        )
        
        bn_var_sqrt = torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data = bn_var_sqrt
        idbn2.bias.data = self.running2.running_mean
        idbn2.running_mean.data = self.running2.running_mean + bias + downsample_bias
        idbn2.running_var.data = self.running2.running_var

        if keep_mask:
            mask2 = nn.Conv2d(self.out_planes, self.out_planes, 1, groups=self.out_planes, bias=False)
            mask2.weight.data = self.mask2.weight.data
        else:
            idbn2.weight.data *= self.mask2.weight.data.reshape(-1)
            idbn2.bias.data *= self.mask2.weight.data.reshape(-1)

        self.reparam = nn.Sequential()
        self.reparam.add_module('conv1', idconv1)
        self.reparam.add_module('bn1', idbn1)
        if keep_mask:
            self.reparam.add_module('mask1', mask1)
        self.reparam.add_module('relu1', nn.ReLU(True))
        self.reparam.add_module('conv2', idconv2)
        self.reparam.add_module('bn2', idbn2)
        if keep_mask:
            self.reparam.add_module('mask2', mask2)
        self.reparam.add_module('relu2', nn.ReLU(True))

        self.__delattr__('conv1')
        self.__delattr__('bn1')
        self.__delattr__('conv2')
        self.__delattr__('bn2')
        self.__delattr__('downsample')

        self.__delattr__('relu')
        self.__delattr__('mask1')
        self.__delattr__('mask2')
        self.__delattr__('mask_res')
        self.__delattr__('running1')
        self.__delattr__('running2')

    def fuse(self,conv_w, bn_rm, bn_rv,bn_w,bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
        return conv_w, conv_b


class RMNetPruning(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_wide=64):
        super(RMNetPruning, self).__init__()
        self.in_planes = base_wide
        self.conv1 = nn.Conv2d(3, base_wide, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_wide)
        self.mask1 = nn.Conv2d(base_wide,base_wide,1,groups=base_wide,bias=False)
        nn.init.ones_(self.mask1.weight)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, base_wide, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_wide * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_wide * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_wide * 8, num_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self.in_planes, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes * 2, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        
        if hasattr(self, 'mask1'):
            out = self.mask1(out)
        
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        
        return out

    def update_mask(self, sr, threshold):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.groups != 1 and m.weight.requires_grad:
                m.weight.grad.data.add_(sr * torch.sign(m.weight.data))
                m1 = m.weight.data.abs() > threshold
                m.weight.grad.data *= m1
                m.weight.data *= m1
    
    def fix_mask(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1,1) and m.groups != 1:
                m.weight.requires_grad = False
                    
    def deploy_model(self):
        model = copy.deepcopy(self)
        
        # merge mask1
        model.bn.weight.data *= model.mask1.weight.data.reshape(-1)
        model.bn.bias.data *= model.mask1.weight.data.reshape(-1)
        model.__delattr__('mask1')

        # merge layers
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        return model
    
    def prune(self):
        def prune_conv_bn(conv, bn, in_mask, out_mask):
            temp_conv = nn.Conv2d(
                int(in_mask.sum()),
                int(out_mask.sum()),
                conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=False
            ).eval()
            temp_conv.weight.data = conv.weight.data[out_mask][:, in_mask]

            temp_bn = nn.BatchNorm2d(int(out_mask.sum())).eval()
            temp_bn.weight.data = bn.weight.data[out_mask]
            temp_bn.bias.data = bn.bias.data[out_mask]
            temp_bn.running_mean = bn.running_mean[out_mask]
            temp_bn.running_var = bn.running_var[out_mask]

            return temp_conv, temp_bn

        in_mask = torch.ones(3) > 0
        model = self.deploy_model()

        prune_cnt = 0; total_cnt = 0

        # first conv
        mask = model.bn1.weight.data.abs().reshape(-1) > 0
        prune_cnt += int(mask.sum())
        total_cnt += int(len(mask))

        temp_conv, temp_bn = prune_conv_bn(model.conv1, model.bn1, in_mask, mask)
        model.conv1 = temp_conv
        model.bn1 = temp_bn
        in_mask = mask

        # other layers
        for m in model.modules():
            if isinstance(m, ResBlock):
                # mask1
                mask1 = m.reparam.bn1.weight.data.abs().reshape(-1) > 0
                prune_cnt += int(mask1.sum())
                total_cnt += int(len(mask1))

                temp_conv, temp_bn = prune_conv_bn(m.reparam.conv1, m.reparam.bn1, in_mask, mask1)
                m.reparam.conv1 = temp_conv
                m.reparam.bn1 = temp_bn
                in_mask = mask1

                # mask2
                mask2 = m.bn2.weight.data.abs().reshape(-1) > 0
                prune_cnt += int(mask2.sum())
                total_cnt += int(len(mask2))

                temp_conv, temp_bn = prune_conv_bn(m.reparam.conv2, m.reparam.bn2, in_mask, mask2)
                m.reparam.conv2 = temp_conv
                m.reparam.bn2 = temp_bn
                in_mask = mask2

        print(f'=> pruning ratio: {1 - prune_cnt / total_cnt}')
        return model


def _rmnet_pruning(arch, block, layers, pretrained, **kwargs):
    model = RMNetPruning(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')), strict=False
        )
        print("success load pretrained model")

    return model


def rmnet_pruning_18(pretrained=False, num_classes=1000):
    return _rmnet_pruning(
        'rmnet_pruning_18', ResBlock, [2, 2, 2, 2],
        pretrained=pretrained, num_classes=num_classes
    )


def rmnet_pruning_34(pretrained=False, num_classes=1000):
    return _rmnet_pruning(
        'rmnet_pruning_34', ResBlock, [3, 4, 6, 3],
        pretrained=pretrained, num_classes=num_classes
    )

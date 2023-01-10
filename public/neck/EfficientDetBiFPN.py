import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import c2_xavier_fill
import math



class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def hard_swish(self, x, inplace):
        inner = F.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if inplace else x.mul(inner)

    def forward(self, x):
        return self.hard_swish(x, self.inplace)


class SeparableConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(inplanes,
                                        inplanes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=inplanes,
                                        bias=False)
        self.pointwise_conv = nn.Conv2d(inplanes,
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)

        return x


class EfficientDetBiFPN(nn.Module):
    def __init__(self,
                 C3_inplanes,
                 C4_inplanes,
                 C5_inplanes,
                 planes,
                 first_time=False,
                 epsilon=1e-4):
        super(EfficientDetBiFPN, self).__init__()
        self.first_time = first_time
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(planes, planes)
        self.conv5_up = SeparableConvBlock(planes, planes)
        self.conv4_up = SeparableConvBlock(planes, planes)
        self.conv3_up = SeparableConvBlock(planes, planes)
        self.conv4_down = SeparableConvBlock(planes, planes)
        self.conv5_down = SeparableConvBlock(planes, planes)
        self.conv6_down = SeparableConvBlock(planes, planes)
        self.conv7_down = SeparableConvBlock(planes, planes)

        self.p4_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p5_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p6_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p7_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.hardswish = HardSwish(inplace=True)

        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                nn.Conv2d(C5_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(C4_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(C3_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(C5_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1), )

            self.p4_down_channel_2 = nn.Sequential(
                nn.Conv2d(C4_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )
            self.p5_down_channel_2 = nn.Sequential(
                nn.Conv2d(C5_inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(planes),
            )

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.first_time:
            [C3, C4, C5] = inputs

            P3 = self.p3_down_channel(C3)
            P4 = self.p4_down_channel(C4)
            P5 = self.p5_down_channel(C5)

            P6 = self.p5_to_p6(C5)
            P7 = self.p6_to_p7(P6)

        else:
            [P3, P4, P5, P6, P7] = inputs
        # P7_0 to P7_2
        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        P6_up = self.conv6_up(
            self.hardswish(weight[0] * P6 + weight[1] * F.interpolate(
                P7, size=(P6.shape[2], P6.shape[3]), mode='nearest')))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        P5_up = self.conv5_up(
            self.hardswish(weight[0] * P5 + weight[1] * F.interpolate(
                P6_up, size=(P5.shape[2], P5.shape[3]), mode='nearest')))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        P4_up = self.conv4_up(
            self.hardswish(weight[0] * P4 + weight[1] * F.interpolate(
                P5_up, size=(P4.shape[2], P4.shape[3]), mode='nearest')))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        P3_out = self.conv3_up(
            self.hardswish(weight[0] * P3 + weight[1] * F.interpolate(
                P4_up, size=(P3.shape[2], P3.shape[3]), mode='nearest')))

        if self.first_time:
            P4 = self.p4_down_channel_2(C4)
            P5 = self.p5_down_channel_2(C5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        P4_out = self.conv4_down(
            self.hardswish(weight[0] * P4 + weight[1] * P4_up +
                           weight[2] * self.p4_downsample(P3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        P5_out = self.conv5_down(
            self.hardswish(weight[0] * P5 + weight[1] * P5_up +
                           weight[2] * self.p5_downsample(P4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        P6_out = self.conv6_down(
            self.hardswish(weight[0] * P6 + weight[1] * P6_up +
                           weight[2] * self.p6_downsample(P5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        P7_out = self.conv7_down(
            self.hardswish(weight[0] * P7 +
                           weight[1] * self.p7_downsample(P6_out)))

        return [P3_out, P4_out, P5_out, P6_out, P7_out]

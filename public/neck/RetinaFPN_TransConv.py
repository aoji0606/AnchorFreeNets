import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import c2_xavier_fill
import math


class RetinaFPN_TransConv(nn.Module):
    def __init__(self,
                 inplanes:list,
                 planes:int,
                 use_p5:bool=True,
                 fpn_bn:bool=True):
        """
        @description  : the normal FPN
        ---------
        @param  :
        inplanes: the channel_num of the stage3 to stage5 in backbone(for example, [512,1024,2048] in resnet50)
        planes: the channel num in up process
        use_p5: use the feature in upper_stage5
        fpn_bn: use bn in after transConv
        -------
        @Returns  :
        feature of stage3 to stage7
        -------
        """
        super(RetinaFPN_TransConv, self).__init__()
        C3_inplanes,C4_inplanes,C5_inplanes = inplanes
        self.use_p5 = use_p5
        self.P3_1 = nn.Conv2d(C3_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        # self.P3_2 = nn.Conv2d(planes,
        #                       planes,
        #                       kernel_size=3,
        #                       stride=1,
        #                       padding=1)
        self.P4_1 = nn.Conv2d(C4_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        # self.P4_2 = nn.Conv2d(planes,
        #                       planes,
        #                       kernel_size=3,
        #                       stride=1,
        #                       padding=1)
        self.P5_1 = nn.Conv2d(C5_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        if self.use_p5:
            self.P6 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        else:
            self.P6 = nn.Conv2d(C5_inplanes,
                                planes,
                                kernel_size=3,
                                stride=2,
                                padding=1)

        self.P7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))
        
        self.up54 = nn.ConvTranspose2d(in_channels=planes, out_channels=planes, kernel_size=4, stride=2, padding=1, bias=False)
        self.up43 = nn.ConvTranspose2d(in_channels=planes, out_channels=planes, kernel_size=4, stride=2, padding=1, bias=False)
        if fpn_bn:
            self.conv54 = nn.Sequential(nn.Conv2d(in_channels=2*planes, out_channels=planes, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(planes, eps=0.0001))
            self.conv43 = nn.Sequential(nn.Conv2d(in_channels=2*planes, out_channels=planes, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(planes, eps=0.0001))
        else:
            self.conv54 = nn.Conv2d(in_channels=2*planes, out_channels=planes, kernel_size=3, padding=1, bias=False)
            self.conv43 = nn.Conv2d(in_channels=2*planes, out_channels=planes, kernel_size=3, padding=1, bias=False)

    def forward(self, inputs):
        C3, C4, C5 = inputs[-3], inputs[-2], inputs[-1]
        del inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4_f_up = self.up54(P5)
        P4_f_up = torch.cat((P4, P4_f_up), dim=1)
        P4 = self.conv54(P4_f_up)

        P3 = self.P3_1(C3)
        P3_f_up = self.up43(P4)
        P3_f_up = torch.cat((P3, P3_f_up), dim=1)
        P3 = self.conv43(P3_f_up)


        P5 = self.P5_2(P5)
        # P4 = self.P4_2(P4)
        # P3 = self.P3_2(P3)

        # if self.use_p5:
            # P6 = self.P6(P5)
        # else:
        P6 = self.P6(P5)

        del C3, C4, C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]
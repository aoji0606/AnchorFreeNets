import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import c2_xavier_fill
import math






class RetinaFPN(nn.Module):
    def __init__(self,
                 inplanes:list,
                 planes:int,
                 use_p5:bool=True,
                 fpn_bn:bool=False):
        """
        @description  : the normal FPN
        ---------
        @param  :
        inplanes: the channel_num of the stage3 to stage5 in backbone(for example, [512,1024,2048] in resnet50)
        planes: the channel num in up process
        use_p5: use the feature in upper_stage5
        -------
        @Returns  :
        feature of stage3 to stage7
        -------
        """
        
        
        super(RetinaFPN, self).__init__()
        C3_inplanes,C4_inplanes,C5_inplanes = inplanes
        self.use_p5 = use_p5
        self.P3_1 = nn.Conv2d(C3_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(C4_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
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

    def forward(self, inputs):
        C3, C4, C5 = inputs[-3], inputs[-2], inputs[-1]
        del inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                           mode='nearest') + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        if self.use_p5:
            P6 = self.P6(P5)
        else:
            P6 = self.P6(C5)

        del C3, C4, C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]
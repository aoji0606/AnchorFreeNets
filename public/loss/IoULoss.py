import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class IoUMethod:
    def __init__(self):
        pass

    def __call__(self, boxes1, boxes2, iou_type='IoU', box_type='xyxy'):
        '''
        box1 format:[...,4]
        box2 format:[...,4]
        xyxy type:[x_min,y_min,x_max,y_max]
        xywh type:[x_center,y_center,w,h]
        '''
        assert iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU'], 'wrong IoU type!'
        assert box_type in ['xyxy', 'xywh'], 'wrong box_type type!'

        if box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[:, 0:2] - boxes1[:, 2:4] / 2
            boxes1_x2y2 = boxes1[:, 0:2] + boxes1[:, 2:4] / 2
            boxes1 = torch.cat([boxes1_x1y1, boxes1_x2y2], dim=1)

            boxes2_x1y1 = boxes2[:, 0:2] - boxes2[:, 2:4] / 2
            boxes2_x2y2 = boxes2[:, 0:2] + boxes2[:, 2:4] / 2
            boxes2 = torch.cat([boxes2_x1y1, boxes2_x2y2], dim=1)

        overlap_area_xymin = torch.max(boxes1[..., 0:2], boxes2[..., 0:2])
        overlap_area_xymax = torch.min(boxes1[..., 2:4], boxes2[..., 2:4])
        overlap_area_sizes = torch.clamp(overlap_area_xymax -
                                         overlap_area_xymin,
                                         min=0)
        overlap_area = overlap_area_sizes[..., 0] * overlap_area_sizes[..., 1]

        boxes1_wh = torch.clamp(boxes1[..., 2:4] - boxes1[..., 0:2], min=0)
        boxes2_wh = torch.clamp(boxes2[..., 2:4] - boxes2[..., 0:2], min=0)

        boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

        # compute ious between boxes1 and boxes2
        union_area = boxes1_area + boxes2_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        ious = overlap_area / union_area

        if iou_type == 'IoU':
            return ious
        else:
            if iou_type in ['GIoU', 'DIoU', 'CIoU']:
                enclose_area_top_left = torch.min(boxes1[..., 0:2],
                                                  boxes2[..., 0:2])
                enclose_area_bot_right = torch.max(boxes1[..., 2:4],
                                                   boxes2[..., 2:4])
                enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                                 enclose_area_top_left,
                                                 min=0)
                if iou_type in ['DIoU', 'CIoU']:
                    # https://arxiv.org/abs/1911.08287v1
                    # compute DIoU c2 and p2
                    # c2:convex diagonal squared
                    c2 = enclose_area_sizes[...,
                                            0]**2 + enclose_area_sizes[...,
                                                                       1]**2
                    c2 = torch.clamp(c2, min=1e-4)
                    # p2:center distance squared
                    boxes1_ctr = (boxes1[..., 2:4] + boxes1[..., 0:2]) / 2
                    boxes2_ctr = (boxes2[..., 2:4] + boxes2[..., 0:2]) / 2
                    p2 = (boxes1_ctr[..., 0] - boxes2_ctr[..., 0])**2 + (
                        boxes1_ctr[..., 1] - boxes2_ctr[..., 1])**2
                    # compute CIoU v and alpha
                    v = (4 / math.pi**2) * torch.pow(
                        torch.atan(boxes2_wh[:, 0] / boxes2_wh[:, 1]) -
                        torch.atan(boxes1_wh[:, 0] / boxes1_wh[:, 1]), 2)
                    with torch.no_grad():
                        alpha = v / torch.clamp(1 - ious + v, min=1e-4)

                    return ious - p2 / c2 if iou_type == 'DIoU' else ious - (
                        p2 / c2 + v * alpha)
                else:
                    enclose_area = enclose_area_sizes[:,
                                                      0] * enclose_area_sizes[:,
                                                                              1]
                    enclose_area = torch.clamp(enclose_area, min=1e-4)

                    return ious - (enclose_area - union_area) / enclose_area
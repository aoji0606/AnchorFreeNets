import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



# class YOLOV5Loss(nn.Module):
#     def __init__(self,
#                  anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
#                                [62, 45], [59, 119], [116, 90], [156, 198],
#                                [373, 326]],
#                  per_level_num_anchors=3,
#                  strides=[8, 16, 32],
#                  epsilon=1e-4):
#         super(YOLOV5Loss, self).__init__()
#         self.anchor_sizes = anchor_sizes
#         self.per_level_num_anchors = per_level_num_anchors
#         self.strides = strides
#         self.epsilon = epsilon

#     def forward(self, obj_heads, reg_heads, cls_heads, batch_anchors,
#                 annotations):
#         """
#         compute obj loss, reg loss and cls loss in one batch
#         """
#         device = annotations.device

#         obj_preds, reg_preds, cls_preds, all_anchors = [], [], [], []
#         for per_level_obj_pred, per_level_reg_pred, per_level_cls_pred, per_level_anchors in zip(
#                 obj_heads, reg_heads, cls_heads, batch_anchors):
#             per_level_obj_pred = per_level_obj_pred.view(
#                 per_level_obj_pred.shape[0], -1, per_level_obj_pred.shape[-1])
#             per_level_reg_pred = per_level_reg_pred.view(
#                 per_level_reg_pred.shape[0], -1, per_level_reg_pred.shape[-1])
#             per_level_cls_pred = per_level_cls_pred.view(
#                 per_level_cls_pred.shape[0], -1, per_level_cls_pred.shape[-1])
#             per_level_anchors = per_level_anchors.view(
#                 per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

#             obj_preds.append(per_level_obj_pred)
#             reg_preds.append(per_level_reg_pred)
#             cls_preds.append(per_level_cls_pred)
#             all_anchors.append(per_level_anchors)

#         obj_preds = torch.cat(obj_preds, axis=1)
#         reg_preds = torch.cat(reg_preds, axis=1)
#         cls_preds = torch.cat(cls_preds, axis=1)
#         all_anchors = torch.cat(all_anchors, axis=1)

#         obj_preds = torch.sigmoid(obj_preds)
#         cls_preds = torch.sigmoid(cls_preds)
#         # snap  reg_preds from tx,ty,tw,th -> x_center,y_center,w,h -> x_min,y_min,x_max,y_max
#         reg_preds[:, :,
#                   0:2] = (torch.sigmoid(reg_preds[:, :, 0:2]) +
#                           all_anchors[:, :, 0:2]) * all_anchors[:, :, 4:5]
#         reg_preds[:, :, 2:4] = torch.exp(
#             reg_preds[:, :, 2:4]) * all_anchors[:, :, 2:4]
#         reg_preds[:, :,
#                   0:2] = reg_preds[:, :, 0:2] - 0.5 * reg_preds[:, :, 2:4]
#         reg_preds[:, :, 2:4] = reg_preds[:, :, 2:4] + reg_preds[:, :, 0:2]
#         print("1111", obj_preds.shape, reg_preds.shape, cls_preds.shape,
#               all_anchors.shape)
#         # batch_anchor_targets = self.get_batch_anchors_targets(
#         #     batch_anchors, annotations)

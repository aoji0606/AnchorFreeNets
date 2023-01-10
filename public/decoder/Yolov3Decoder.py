import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class YOLOV3Decoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 top_n=1000,
                 min_score_threshold=0.05,
                 nms_threshold=0.5,
                 max_detection_num=100):
        super(YOLOV3Decoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.top_n = top_n
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num

    def forward(self, obj_heads, reg_heads, cls_heads, batch_anchors):
        device = cls_heads[0].device
        with torch.no_grad():
            filter_scores, filter_score_classes, filter_boxes = [], [], []
            for per_level_obj_pred, per_level_reg_pred, per_level_cls_pred, per_level_anchors in zip(
                    obj_heads, reg_heads, cls_heads, batch_anchors):
                per_level_obj_pred = per_level_obj_pred.view(
                    per_level_obj_pred.shape[0], -1,
                    per_level_obj_pred.shape[-1])
                per_level_reg_pred = per_level_reg_pred.view(
                    per_level_reg_pred.shape[0], -1,
                    per_level_reg_pred.shape[-1])
                per_level_cls_pred = per_level_cls_pred.view(
                    per_level_cls_pred.shape[0], -1,
                    per_level_cls_pred.shape[-1])
                per_level_anchors = per_level_anchors.view(
                    per_level_anchors.shape[0], -1,
                    per_level_anchors.shape[-1])

                per_level_obj_pred = torch.sigmoid(per_level_obj_pred)
                per_level_cls_pred = torch.sigmoid(per_level_cls_pred)

                # snap per_level_reg_pred from tx,ty,tw,th -> x_center,y_center,w,h -> x_min,y_min,x_max,y_max
                per_level_reg_pred[:, :, 0:2] = (
                    torch.sigmoid(per_level_reg_pred[:, :, 0:2]) +
                    per_level_anchors[:, :, 0:2]) * per_level_anchors[:, :,
                                                                      4:5]
                # pred_bboxes_wh=exp(twh)*anchor_wh/stride
                per_level_reg_pred[:, :, 2:4] = torch.exp(
                    per_level_reg_pred[:, :, 2:4]
                ) * per_level_anchors[:, :, 2:4] / per_level_anchors[:, :, 4:5]

                per_level_reg_pred[:, :, 0:
                                   2] = per_level_reg_pred[:, :, 0:
                                                           2] - 0.5 * per_level_reg_pred[:, :,
                                                                                         2:
                                                                                         4]
                per_level_reg_pred[:, :, 2:
                                   4] = per_level_reg_pred[:, :, 2:
                                                           4] + per_level_reg_pred[:, :,
                                                                                   0:
                                                                                   2]
                per_level_reg_pred = per_level_reg_pred.int()
                per_level_reg_pred[:, :,
                                   0] = torch.clamp(per_level_reg_pred[:, :,
                                                                       0],
                                                    min=0)
                per_level_reg_pred[:, :,
                                   1] = torch.clamp(per_level_reg_pred[:, :,
                                                                       1],
                                                    min=0)
                per_level_reg_pred[:, :,
                                   2] = torch.clamp(per_level_reg_pred[:, :,
                                                                       2],
                                                    max=self.image_w - 1)
                per_level_reg_pred[:, :,
                                   3] = torch.clamp(per_level_reg_pred[:, :,
                                                                       3],
                                                    max=self.image_h - 1)

                per_level_scores, per_level_score_classes = torch.max(
                    per_level_cls_pred, dim=2)
                per_level_scores = per_level_scores * per_level_obj_pred.squeeze(
                    -1)
                if per_level_scores.shape[1] >= self.top_n:
                    per_level_scores, indexes = torch.topk(per_level_scores,
                                                           self.top_n,
                                                           dim=1,
                                                           largest=True,
                                                           sorted=True)
                    per_level_score_classes = torch.gather(
                        per_level_score_classes, 1, indexes)
                    per_level_reg_pred = torch.gather(
                        per_level_reg_pred, 1,
                        indexes.unsqueeze(-1).repeat(1, 1, 4))

                filter_scores.append(per_level_scores)
                filter_score_classes.append(per_level_score_classes)
                filter_boxes.append(per_level_reg_pred)

            filter_scores = torch.cat(filter_scores, axis=1)
            filter_score_classes = torch.cat(filter_score_classes, axis=1)
            filter_boxes = torch.cat(filter_boxes, axis=1)

            batch_scores, batch_classes, batch_pred_bboxes = [], [], []
            for scores, score_classes, pred_bboxes in zip(
                    filter_scores, filter_score_classes, filter_boxes):
                score_classes = score_classes[
                    scores > self.min_score_threshold].float()
                pred_bboxes = pred_bboxes[
                    scores > self.min_score_threshold].float()
                scores = scores[scores > self.min_score_threshold].float()

                one_image_scores = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_classes = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_pred_bboxes = (-1) * torch.ones(
                    (self.max_detection_num, 4), device=device)

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_pred_bboxes = pred_bboxes[sorted_indexes]

                    keep = nms(sorted_pred_bboxes, sorted_scores,
                               self.nms_threshold)
                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_pred_bboxes = sorted_pred_bboxes[keep]

                    final_detection_num = min(self.max_detection_num,
                                              keep_scores.shape[0])

                    one_image_scores[0:final_detection_num] = keep_scores[
                        0:final_detection_num]
                    one_image_classes[0:final_detection_num] = keep_classes[
                        0:final_detection_num]
                    one_image_pred_bboxes[
                        0:final_detection_num, :] = keep_pred_bboxes[
                            0:final_detection_num, :]

                one_image_scores = one_image_scores.unsqueeze(0)
                one_image_classes = one_image_classes.unsqueeze(0)
                one_image_pred_bboxes = one_image_pred_bboxes.unsqueeze(0)

                batch_scores.append(one_image_scores)
                batch_classes.append(one_image_classes)
                batch_pred_bboxes.append(one_image_pred_bboxes)

            batch_scores = torch.cat(batch_scores, axis=0)
            batch_classes = torch.cat(batch_classes, axis=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            return batch_scores, batch_classes, batch_pred_bboxes
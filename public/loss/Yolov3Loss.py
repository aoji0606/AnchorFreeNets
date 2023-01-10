import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class YOLOV3Loss(nn.Module):
    def __init__(self,
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 per_level_num_anchors=3,
                 strides=[8, 16, 32],
                 obj_weight=1.0,
                 noobj_weight=100.0,
                 epsilon=1e-4):
        super(YOLOV3Loss, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.per_level_num_anchors = per_level_num_anchors
        self.strides = strides
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.epsilon = epsilon

    def forward(self, obj_heads, reg_heads, cls_heads, batch_anchors,
                annotations):
        """
        compute obj loss, reg loss and cls loss in one batch
        """
        device = annotations.device
        batch_anchor_targets = self.get_batch_anchors_targets(
            batch_anchors, annotations)

        obj_noobj_loss = torch.tensor(0.).to(device)
        reg_loss = torch.tensor(0.).to(device)
        cls_loss = torch.tensor(0.).to(device)
        for per_level_obj_pred, per_level_reg_pred, per_level_cls_pred, per_level_anchors, per_level_anchor_targets in zip(
                obj_heads, reg_heads, cls_heads, batch_anchors,
                batch_anchor_targets):
            per_level_obj_pred = per_level_obj_pred.view(
                per_level_obj_pred.shape[0], -1, per_level_obj_pred.shape[-1])
            per_level_reg_pred = per_level_reg_pred.view(
                per_level_reg_pred.shape[0], -1, per_level_reg_pred.shape[-1])
            per_level_cls_pred = per_level_cls_pred.view(
                per_level_cls_pred.shape[0], -1, per_level_cls_pred.shape[-1])
            per_level_anchors = per_level_anchors.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

            per_level_obj_pred = torch.sigmoid(per_level_obj_pred)
            per_level_cls_pred = torch.sigmoid(per_level_cls_pred)

            # # snap per_level_reg_pred from tx,ty,tw,th -> x_center,y_center,w,h -> x_min,y_min,x_max,y_max
            # per_level_reg_pred[:, :, 0:2] = (
            #     torch.sigmoid(per_level_reg_pred[:, :, 0:2]) +
            #     per_level_anchors[:, :, 0:2]) * per_level_anchors[:, :, 4:5]
            # # pred_bboxes_wh=exp(twh)*anchor_wh/stride
            # per_level_reg_pred[:, :, 2:4] = torch.exp(
            #     per_level_reg_pred[:, :, 2:4]
            # ) * per_level_anchors[:, :, 2:4] / per_level_anchors[:, :, 4:5]

            # per_level_reg_pred[:, :, 0:
            #                    2] = per_level_reg_pred[:, :, 0:
            #                                            2] - 0.5 * per_level_reg_pred[:, :,
            #                                                                          2:
            #                                                                          4]
            # per_level_reg_pred[:, :, 2:
            #                    4] = per_level_reg_pred[:, :, 2:
            #                                            4] + per_level_reg_pred[:, :,
            #                                                                    0:
            #                                                                    2]

            # per_level_anchor_targets[:, :, 0:2] = (
            #     per_level_anchor_targets[:, :, 0:2] +
            #     per_level_anchors[:, :, 0:2]) * per_level_anchors[:, :, 4:5]
            # per_level_anchor_targets[:, :, 2:4] = torch.exp(
            #     per_level_anchor_targets[:, :, 2:4]
            # ) * per_level_anchors[:, :, 2:4] / per_level_anchors[:, :, 4:5]
            # per_level_anchor_targets[:, :, 0:
            #                          2] = per_level_anchor_targets[:, :, 0:
            #                                                        2] - 0.5 * per_level_anchor_targets[:, :,
            #                                                                                            2:
            #                                                                                            4]
            # per_level_anchor_targets[:, :, 2:
            #                          4] = per_level_anchor_targets[:, :, 2:
            #                                                        4] + per_level_anchor_targets[:, :,
            #                                                                                      0:
            #                                                                                      2]

            per_level_reg_pred = per_level_reg_pred.type_as(per_level_obj_pred)
            per_level_cls_pred = per_level_cls_pred.type_as(per_level_obj_pred)
            per_level_anchors = per_level_anchors.type_as(per_level_obj_pred)
            per_level_anchor_targets = per_level_anchor_targets.type_as(
                per_level_obj_pred)

            per_level_obj_pred = per_level_obj_pred.view(
                -1, per_level_obj_pred.shape[-1])
            per_level_reg_pred = per_level_reg_pred.view(
                -1, per_level_reg_pred.shape[-1])
            per_level_cls_pred = per_level_cls_pred.view(
                -1, per_level_cls_pred.shape[-1])
            per_level_anchor_targets = per_level_anchor_targets.view(
                -1, per_level_anchor_targets.shape[-1])

            obj_noobj_loss = obj_noobj_loss + self.compute_per_level_batch_obj_noobj_loss(
                per_level_obj_pred, per_level_anchor_targets)
            reg_loss = reg_loss + self.compute_per_level_batch_reg_loss(
                per_level_reg_pred, per_level_anchor_targets)
            cls_loss = cls_loss + self.compute_per_level_batch_cls_loss(
                per_level_cls_pred, per_level_anchor_targets)

        return obj_noobj_loss, reg_loss, cls_loss

    def compute_per_level_batch_obj_noobj_loss(self, per_level_obj_pred,
                                               per_level_anchor_targets):
        """
        compute per level batch obj noobj loss(bce loss)
        per_level_obj_pred:[batch_size*per_level_anchor_num,1]
        per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
        """
        device = per_level_obj_pred.device
        positive_anchors_num = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0].shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        positive_obj_preds = per_level_obj_pred[
            per_level_anchor_targets[:, 5] > 0].view(-1)
        positive_obj_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0][:, 5:6].view(-1)

        negative_obj_preds = (
            1. -
            per_level_obj_pred[per_level_anchor_targets[:, 6] > 0]).view(-1)
        negative_obj_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 6] > 0][:, 6:7].view(-1)

        obj_loss = -(positive_obj_targets * torch.log(positive_obj_preds))
        noobj_loss = -(negative_obj_targets * torch.log(negative_obj_preds))

        obj_loss = obj_loss.mean()
        noobj_loss = noobj_loss.mean()
        total_loss = self.obj_weight * obj_loss + self.noobj_weight * noobj_loss

        return total_loss

    def compute_per_level_batch_reg_loss(self, per_level_reg_pred,
                                         per_level_anchor_targets):
        """
        compute per level batch reg loss(mse loss)
        per_level_reg_pred:[batch_size*per_level_anchor_num,4]
        per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
        """
        # only use positive anchor sample to compute reg loss
        device = per_level_reg_pred.device
        per_level_reg_pred = per_level_reg_pred[
            per_level_anchor_targets[:, 5] > 0]
        per_level_reg_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0][:, 0:4]

        positive_anchors_num = per_level_reg_targets.shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        reg_loss = (per_level_reg_pred - per_level_reg_targets)**2
        reg_loss = reg_loss.sum(axis=1)

        reg_loss = reg_loss.mean()

        return reg_loss

    # def compute_per_level_batch_reg_loss(self, per_level_reg_pred,
    #                                      per_level_anchor_targets):
    #     """
    #     compute per level batch reg loss(giou loss)
    #     per_level_reg_pred:[batch_size*per_level_anchor_num,4]
    #     per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
    #     """
    #     # only use positive anchor sample to compute reg loss
    #     device = per_level_reg_pred.device
    #     per_level_reg_pred = per_level_reg_pred[
    #         per_level_anchor_targets[:, 5] > 0]
    #     per_level_reg_targets = per_level_anchor_targets[
    #         per_level_anchor_targets[:, 5] > 0][:, 0:4]

    #     positive_anchors_num = per_level_reg_targets.shape[0]

    #     if positive_anchors_num == 0:
    #         return torch.tensor(0.).to(device)

    #     overlap_area_top_left = torch.max(per_level_reg_pred[:, 0:2],
    #                                       per_level_reg_targets[:, 0:2])
    #     overlap_area_bot_right = torch.min(per_level_reg_pred[:, 2:4],
    #                                        per_level_reg_targets[:, 2:4])
    #     overlap_area_sizes = torch.clamp(overlap_area_bot_right -
    #                                      overlap_area_top_left,
    #                                      min=0)
    #     overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

    #     # anchors and annotations convert format to [x1,y1,w,h]
    #     pred_bboxes_w_h = per_level_reg_pred[:, 2:4] - per_level_reg_pred[:,
    #                                                                       0:2]
    #     annotations_w_h = per_level_reg_targets[:, 2:
    #                                             4] - per_level_reg_targets[:,
    #                                                                        0:2]
    #     pred_bboxes_w_h = torch.clamp(pred_bboxes_w_h, min=0)
    #     annotations_w_h = torch.clamp(annotations_w_h, min=0)
    #     # compute anchors_area and annotations_area
    #     pred_bboxes_area = pred_bboxes_w_h[:, 0] * pred_bboxes_w_h[:, 1]
    #     annotations_area = annotations_w_h[:, 0] * annotations_w_h[:, 1]

    #     # compute union_area
    #     union_area = pred_bboxes_area + annotations_area - overlap_area
    #     union_area = torch.clamp(union_area, min=1e-4)
    #     # compute ious between one image anchors and one image annotations
    #     ious = overlap_area / union_area

    #     enclose_area_top_left = torch.min(per_level_reg_pred[:, 0:2],
    #                                       per_level_reg_targets[:, 0:2])
    #     enclose_area_bot_right = torch.max(per_level_reg_pred[:, 2:4],
    #                                        per_level_reg_targets[:, 2:4])
    #     enclose_area_sizes = torch.clamp(enclose_area_bot_right -
    #                                      enclose_area_top_left,
    #                                      min=0)
    #     enclose_area = enclose_area_sizes[:, 0] * enclose_area_sizes[:, 1]
    #     enclose_area = torch.clamp(enclose_area, min=1e-4)

    #     gious_loss = 1. - ious + (enclose_area - union_area) / enclose_area
    #     gious_loss = gious_loss.sum() / positive_anchors_num

    #     return gious_loss

    def compute_per_level_batch_cls_loss(self, per_level_cls_pred,
                                         per_level_anchor_targets):
        """
        compute per level batch cls loss(bce loss)
        per_level_cls_pred:[batch_size*per_level_anchor_num,num_classes]
        per_level_anchor_targets:[batch_size*per_level_anchor_num,7]
        """
        device = per_level_cls_pred.device
        per_level_cls_pred = per_level_cls_pred[
            per_level_anchor_targets[:, 5] > 0]
        per_level_cls_pred = torch.clamp(per_level_cls_pred,
                                         min=self.epsilon,
                                         max=1. - self.epsilon)
        cls_targets = per_level_anchor_targets[
            per_level_anchor_targets[:, 5] > 0][:, 4]

        positive_anchors_num = cls_targets.shape[0]
        num_classes = per_level_cls_pred.shape[1]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(cls_targets.long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        cls_loss = -(
            loss_ground_truth * torch.log(per_level_cls_pred) +
            (1. - loss_ground_truth) * torch.log(1. - per_level_cls_pred))
        cls_loss = cls_loss.sum(axis=1)

        cls_loss = cls_loss.mean()

        return cls_loss

    def get_batch_anchors_targets(self, batch_anchors, annotations):
        """
        Assign a ground truth target for each anchor
        """
        device = annotations.device

        self.anchor_sizes = torch.tensor(self.anchor_sizes,
                                         dtype=torch.float).to(device)
        anchor_sizes = self.anchor_sizes.view(
            self.per_level_num_anchors,
            len(self.anchor_sizes) // self.per_level_num_anchors, 2)

        anchor_level_feature_map_hw = []
        for per_level_anchor in batch_anchors:
            _, H, W, _, _ = per_level_anchor.shape
            anchor_level_feature_map_hw.append([H, W])
        anchor_level_feature_map_hw = torch.tensor(
            anchor_level_feature_map_hw).to(device)

        per_grid_relative_index = []
        for i in range(self.per_level_num_anchors):
            per_grid_relative_index.append(i)
        per_grid_relative_index = torch.tensor(per_grid_relative_index).to(
            device)

        batch_anchor_targets = []
        for per_level_anchor_sizes, stride, per_level_anchors in zip(
                anchor_sizes, self.strides, batch_anchors):
            B, H, W, _, _ = per_level_anchors.shape
            per_level_reg_cls_target = torch.ones(
                [B, H, W, self.per_level_num_anchors, 5], device=device) * (-1)
            # noobj mask init value=0
            per_level_obj_mask = torch.zeros(
                [B, H, W, self.per_level_num_anchors, 1], device=device)
            # noobj mask init value=1
            per_level_noobj_mask = torch.ones(
                [B, H, W, self.per_level_num_anchors, 1], device=device)
            # 7:[x_min,y_min,x_max,y_max,class_label,obj_mask,noobj_mask]
            per_level_anchor_targets = torch.cat([
                per_level_reg_cls_target, per_level_obj_mask,
                per_level_noobj_mask
            ],
                                                 axis=-1)
            per_level_anchor_targets = per_level_anchor_targets.view(
                per_level_anchor_targets.shape[0], -1,
                per_level_anchor_targets.shape[-1])
            per_level_anchors = per_level_anchors.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

            for image_index, one_image_annotations in enumerate(annotations):
                # drop all index=-1 class annotations
                one_image_annotations = one_image_annotations[
                    one_image_annotations[:, 4] >= 0]

                if one_image_annotations.shape[0] != 0:
                    one_image_gt_boxes = one_image_annotations[:, 0:4]
                    one_image_gt_classes = one_image_annotations[:, 4]
                    one_image_gt_boxes_ctr = (one_image_gt_boxes[:, 0:2] +
                                              one_image_gt_boxes[:, 2:4]) / 2

                    # compute all annotations center_grid_index
                    grid_y_indexes = one_image_gt_boxes_ctr[:, 1] // stride
                    grid_x_indexes = one_image_gt_boxes_ctr[:, 0] // stride

                    # compute all annotations gird_indexes transform
                    anchor_indexes_transform = (
                        ((grid_y_indexes * W + grid_x_indexes - 1) *
                         self.per_level_num_anchors).unsqueeze(-1) +
                        per_grid_relative_index.unsqueeze(0)).view(-1)

                    one_image_ious = self.compute_ious_for_one_image(
                        per_level_anchor_sizes, one_image_annotations)

                    # negative anchor includes all anchors with iou <0.5, but the max iou anchor of each annot is not included
                    negative_anchor_flags = (torch.ge(
                        one_image_ious.permute(1, 0), 0.5)).view(-1)
                    negative_anchor_indexes_transform = anchor_indexes_transform[
                        negative_anchor_flags].long()
                    # for anchors which ious>=0.5(ignore threshold),assign noobj_mask label to 0(init value=1)
                    per_level_anchor_targets[image_index,
                                             negative_anchor_indexes_transform,
                                             6] = 0

                    # assign positive sample for max iou anchor of each annot
                    _, positive_anchor_indices = one_image_ious.permute(
                        1, 0).max(axis=1)
                    positive_anchor_indexes_mask = F.one_hot(
                        positive_anchor_indices,
                        num_classes=per_level_anchor_sizes.shape[0]).bool()
                    positive_anchor_indexes_mask = positive_anchor_indexes_mask.view(
                        -1)
                    positive_anchor_indexes_transform = anchor_indexes_transform[
                        positive_anchor_indexes_mask].long()

                    # for positive anchor,assign obj_mask label to 1(init value=0)
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             5] = 1
                    # for positive anchor,assign noobj_mask label to 0(init value=1)
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             6] = 0
                    # for positive anchor,assign class_label:range from 1 to 80
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             4] = one_image_gt_classes + 1
                    # for positive anchor,assign regression_label:[tx,ty,tw,th]
                    per_level_anchor_targets[image_index,
                                             positive_anchor_indexes_transform,
                                             0:2] = (one_image_gt_boxes_ctr %
                                                     stride) / stride
                    one_image_gt_whs = one_image_gt_boxes[:, 2:
                                                          4] - one_image_gt_boxes[:,
                                                                                  0:
                                                                                  2]
                    per_level_anchor_targets[
                        image_index, positive_anchor_indexes_transform,
                        2:4] = torch.log((one_image_gt_whs.float() / (
                            (per_level_anchors[
                                image_index, positive_anchor_indexes_transform,
                                2:4]).float() / stride)) + self.epsilon)

                    judge_positive_anchors = ((one_image_gt_whs.float() / (
                        (per_level_anchors[image_index,
                                           positive_anchor_indexes_transform,
                                           2:4]).float() / stride)) >= 1.)
                    judge_flags = ((judge_positive_anchors[:, 0].int() +
                                    judge_positive_anchors[:, 1].int()) < 2)
                    illegal_anchor_mask = []
                    for flag in judge_flags:
                        for _ in range(self.per_level_num_anchors):
                            illegal_anchor_mask.append(flag)
                    illegal_anchor_mask = torch.tensor(illegal_anchor_mask).to(
                        device)
                    illegal_positive_anchor_indexes_transform = anchor_indexes_transform[
                        illegal_anchor_mask].long()

                    per_level_anchor_targets[
                        image_index, illegal_positive_anchor_indexes_transform,
                        0:5] = -1
                    per_level_anchor_targets[
                        image_index, illegal_positive_anchor_indexes_transform,
                        5] = 0
                    per_level_anchor_targets[
                        image_index, illegal_positive_anchor_indexes_transform,
                        6] = 1

            batch_anchor_targets.append(per_level_anchor_targets)

        return batch_anchor_targets

    def compute_ious_for_one_image(self, anchor_sizes, one_image_annotations):
        """
        compute ious between one image anchors and one image annotations
        """
        annotations_wh = one_image_annotations[:, 2:
                                               4] - one_image_annotations[:,
                                                                          0:2]

        # anchor_sizes format:[anchor_nums,4],4:[anchor_w,anchor_h]
        # annotations_wh format: [annotation_nums,4],4:[gt_w,gt_h]
        # When calculating iou, the upper left corner of anchor_sizes and annotations_wh are point (0, 0)

        anchor_sizes = torch.clamp(anchor_sizes, min=0)
        annotations_wh = torch.clamp(annotations_wh, min=0)
        anchor_areas = anchor_sizes[:, 0] * anchor_sizes[:, 1]
        annotations_areas = annotations_wh[:, 0] * annotations_wh[:, 1]

        overlap_areas_w = torch.min(anchor_sizes[:, 0].unsqueeze(-1),
                                    annotations_wh[:, 0].unsqueeze(0))
        overlap_areas_h = torch.min(anchor_sizes[:, 1].unsqueeze(-1),
                                    annotations_wh[:, 1].unsqueeze(0))
        overlap_areas_w = torch.clamp(overlap_areas_w, min=0)
        overlap_areas_h = torch.clamp(overlap_areas_h, min=0)
        overlap_areas = overlap_areas_w * overlap_areas_h

        union_areas = anchor_areas.unsqueeze(-1) + annotations_areas.unsqueeze(
            0) - overlap_areas
        union_areas = torch.clamp(union_areas, min=1e-4)
        # compute ious between one image anchors and one image annotations
        one_image_ious = (overlap_areas / union_areas)

        return one_image_ious
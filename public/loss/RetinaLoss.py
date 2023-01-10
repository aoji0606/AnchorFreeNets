import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaLoss(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 epsilon=1e-4):
        super(RetinaLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.image_w = image_w
        self.image_h = image_h

    def forward(self, cls_heads, reg_heads, batch_anchors, annotations):
        """
        compute cls loss and reg loss in one batch
        """
        device = annotations.device
        cls_heads = torch.cat(cls_heads, axis=1)
        reg_heads = torch.cat(reg_heads, axis=1)
        batch_anchors = torch.cat(batch_anchors, axis=1)

        cls_heads, reg_heads, batch_anchors = self.drop_out_border_anchors_and_heads(
            cls_heads, reg_heads, batch_anchors, self.image_w, self.image_h)
        batch_anchors_annotations = self.get_batch_anchors_annotations(
            batch_anchors, annotations)

        reg_heads = reg_heads.type_as(cls_heads)
        batch_anchors = batch_anchors.type_as(cls_heads)
        batch_anchors_annotations = batch_anchors_annotations.type_as(
            cls_heads)

        cls_heads = cls_heads.view(-1, cls_heads.shape[-1])
        reg_heads = reg_heads.view(-1, reg_heads.shape[-1])
        batch_anchors_annotations = batch_anchors_annotations.view(
            -1, batch_anchors_annotations.shape[-1])

        positive_anchors_num = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchors_num > 0:
            cls_loss = self.compute_batch_focal_loss(
                cls_heads, batch_anchors_annotations)
            reg_loss = self.compute_batch_smoothl1_loss(
                reg_heads, batch_anchors_annotations)
        else:
            cls_loss = torch.tensor(0.).to(device)
            reg_loss = torch.tensor(0.).to(device)

        return cls_loss, reg_loss

    def compute_batch_focal_loss(self, cls_heads, batch_anchors_annotations):
        """
        compute batch focal loss(cls loss)
        cls_heads:[batch_size*anchor_num,num_classes]
        batch_anchors_annotations:[batch_size*anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        device = cls_heads.device
        cls_heads = cls_heads[batch_anchors_annotations[:, 4] >= 0]
        batch_anchors_annotations = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] >= 0]
        positive_anchors_num = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        cls_heads = torch.clamp(cls_heads,
                                min=self.epsilon,
                                max=1. - self.epsilon)
        num_classes = cls_heads.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_anchors_annotations[:, 4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(cls_heads) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), cls_heads,
                         1. - cls_heads)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        batch_bce_loss = -(
            loss_ground_truth * torch.log(cls_heads) +
            (1. - loss_ground_truth) * torch.log(1. - cls_heads))

        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / positive_anchors_num

        return batch_focal_loss

    def compute_batch_smoothl1_loss(self, reg_heads,
                                    batch_anchors_annotations):
        """
        compute batch smoothl1 loss(reg loss)
        per_image_reg_heads:[batch_size*anchor_num,4]
        per_image_anchors_annotations:[batch_size*anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate smoothl1 loss
        device = reg_heads.device
        reg_heads = reg_heads[batch_anchors_annotations[:, 4] > 0]
        batch_anchors_annotations = batch_anchors_annotations[
            batch_anchors_annotations[:, 4] > 0]
        positive_anchor_num = batch_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smoothl1 loss
        loss_ground_truth = batch_anchors_annotations[:, 0:4]

        x = torch.abs(reg_heads - loss_ground_truth)
        batch_smoothl1_loss = torch.where(torch.ge(x, self.beta),
                                          x - 0.5 * self.beta,
                                          0.5 * (x**2) / self.beta)
        batch_smoothl1_loss = batch_smoothl1_loss.mean(axis=1).sum()
        # according to the original paper,We divide the smoothl1 loss by the number of positive sample anchors
        batch_smoothl1_loss = batch_smoothl1_loss / positive_anchor_num

        return batch_smoothl1_loss

    def drop_out_border_anchors_and_heads(self, cls_heads, reg_heads,
                                          batch_anchors, image_w, image_h):
        """
        dropout out of border anchors,cls heads and reg heads
        """
        final_cls_heads, final_reg_heads, final_batch_anchors = [], [], []
        for per_image_cls_head, per_image_reg_head, per_image_anchors in zip(
                cls_heads, reg_heads, batch_anchors):
            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                                      0] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                                      0] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                                    0] > 0.0]

            per_image_cls_head = per_image_cls_head[per_image_anchors[:,
                                                                      1] > 0.0]
            per_image_reg_head = per_image_reg_head[per_image_anchors[:,
                                                                      1] > 0.0]
            per_image_anchors = per_image_anchors[per_image_anchors[:,
                                                                    1] > 0.0]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 2] < image_w]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 2] < image_w]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 2] < image_w]

            per_image_cls_head = per_image_cls_head[
                per_image_anchors[:, 3] < image_h]
            per_image_reg_head = per_image_reg_head[
                per_image_anchors[:, 3] < image_h]
            per_image_anchors = per_image_anchors[
                per_image_anchors[:, 3] < image_h]

            per_image_cls_head = per_image_cls_head.unsqueeze(0)
            per_image_reg_head = per_image_reg_head.unsqueeze(0)
            per_image_anchors = per_image_anchors.unsqueeze(0)

            final_cls_heads.append(per_image_cls_head)
            final_reg_heads.append(per_image_reg_head)
            final_batch_anchors.append(per_image_anchors)

        final_cls_heads = torch.cat(final_cls_heads, axis=0)
        final_reg_heads = torch.cat(final_reg_heads, axis=0)
        final_batch_anchors = torch.cat(final_batch_anchors, axis=0)

        # final cls heads shape:[batch_size, anchor_nums, class_num]
        # final reg heads shape:[batch_size, anchor_nums, 4]
        # final batch anchors shape:[batch_size, anchor_nums, 4]
        return final_cls_heads, final_reg_heads, final_batch_anchors

    def get_batch_anchors_annotations(self, batch_anchors, annotations):
        """
        Assign a ground truth box target and a ground truth class target for each anchor
        if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
        if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
        if anchor gt_class index > 0,this anchor is a object class anchor and used in
        calculate cls loss and reg loss
        """
        device = annotations.device
        assert batch_anchors.shape[0] == annotations.shape[0]
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchors_annotations = []
        for one_image_anchors, one_image_annotations in zip(
                batch_anchors, annotations):
            # drop all index=-1 class annotations
            one_image_annotations = one_image_annotations[
                one_image_annotations[:, 4] >= 0]

            if one_image_annotations.shape[0] == 0:
                one_image_anchor_annotations = torch.ones(
                    [one_image_anchor_nums, 5], device=device) * (-1)
            else:
                one_image_gt_bboxes = one_image_annotations[:, 0:4]
                one_image_gt_class = one_image_annotations[:, 4]
                one_image_ious = self.compute_ious_for_one_image(
                    one_image_anchors, one_image_gt_bboxes)

                # snap per gt bboxes to the best iou anchor
                overlap, indices = one_image_ious.max(axis=1)
                # assgin each anchor gt bboxes for max iou annotation
                per_image_anchors_gt_bboxes = one_image_gt_bboxes[indices]
                # transform gt bboxes to [tx,ty,tw,th] format for each anchor
                one_image_anchors_snaped_boxes = self.snap_annotations_as_tx_ty_tw_th(
                    per_image_anchors_gt_bboxes, one_image_anchors)

                one_image_anchors_gt_class = (torch.ones_like(overlap) *
                                              -1).to(device)
                # if iou <0.4,assign anchors gt class as 0:background
                one_image_anchors_gt_class[overlap < 0.4] = 0
                # if iou >=0.5,assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 80
                one_image_anchors_gt_class[
                    overlap >=
                    0.5] = one_image_gt_class[indices][overlap >= 0.5] + 1

                one_image_anchors_gt_class = one_image_anchors_gt_class.unsqueeze(
                    -1)

                one_image_anchor_annotations = torch.cat([
                    one_image_anchors_snaped_boxes, one_image_anchors_gt_class
                ],
                                                         axis=1)
            one_image_anchor_annotations = one_image_anchor_annotations.unsqueeze(
                0)
            batch_anchors_annotations.append(one_image_anchor_annotations)

        batch_anchors_annotations = torch.cat(batch_anchors_annotations,
                                              axis=0)

        # batch anchors annotations shape:[batch_size, anchor_nums, 5]
        return batch_anchors_annotations

    def snap_annotations_as_tx_ty_tw_th(self, anchors_gt_bboxes, anchors):
        """
        snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
        """
        anchors_w_h = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_w_h

        anchors_gt_bboxes_w_h = anchors_gt_bboxes[:,
                                                  2:] - anchors_gt_bboxes[:, :2]
        anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1.0)
        anchors_gt_bboxes_ctr = anchors_gt_bboxes[:, :
                                                  2] + 0.5 * anchors_gt_bboxes_w_h

        snaped_annotations_for_anchors = torch.cat(
            [(anchors_gt_bboxes_ctr - anchors_ctr) / anchors_w_h,
             torch.log(anchors_gt_bboxes_w_h / anchors_w_h)],
            axis=1)
        device = snaped_annotations_for_anchors.device
        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        snaped_annotations_for_anchors = snaped_annotations_for_anchors / factor

        # snaped_annotations_for_anchors shape:[anchor_nums, 4]
        return snaped_annotations_for_anchors

    def compute_ious_for_one_image(self, one_image_anchors,
                                   one_image_annotations):
        """
        compute ious between one image anchors and one image annotations
        """
        # make sure anchors format:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        # make sure annotations format: [annotation_nums,4],4:[x_min,y_min,x_max,y_max]
        overlap_area_x_min = torch.max(
            one_image_anchors[:, 0].unsqueeze(-1),
            one_image_annotations[:, 0].unsqueeze(0))
        overlap_area_y_min = torch.max(
            one_image_anchors[:, 1].unsqueeze(-1),
            one_image_annotations[:, 1].unsqueeze(0))
        overlap_area_x_max = torch.min(
            one_image_anchors[:, 2].unsqueeze(-1),
            one_image_annotations[:, 2].unsqueeze(0))
        overlap_area_y_max = torch.min(
            one_image_anchors[:, 3].unsqueeze(-1),
            one_image_annotations[:, 3].unsqueeze(0))
        overlap_areas_w = torch.clamp(overlap_area_x_max - overlap_area_x_min,
                                      min=0)
        overlap_areas_h = torch.clamp(overlap_area_y_max - overlap_area_y_min,
                                      min=0)
        overlaps_area = overlap_areas_w * overlap_areas_h

        anchors_w_h = one_image_anchors[:, 2:] - one_image_anchors[:, :2]
        annotations_w_h = one_image_annotations[:,
                                                2:] - one_image_annotations[:, :
                                                                            2]
        anchors_w_h = torch.clamp(anchors_w_h, min=0)
        annotations_w_h = torch.clamp(annotations_w_h, min=0)
        anchors_area = anchors_w_h[:, 0] * anchors_w_h[:, 1]
        annotations_area = annotations_w_h[:, 0] * annotations_w_h[:, 1]

        unions_area = anchors_area.unsqueeze(-1) + annotations_area.unsqueeze(
            0) - overlaps_area
        unions_area = torch.clamp(unions_area, min=1e-4)
        # compute ious between one image anchors and one image annotations
        one_image_ious = (overlaps_area / unions_area)

        return one_image_ious
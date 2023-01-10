import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



INF = 100000000


class FCOSLoss(nn.Module):
    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]],
                 alpha=0.25,
                 gamma=2.,
                 reg_weight=2.,
                 epsilon=1e-4,
                 center_sample_radius=1.5,
                 use_center_sample=True):
        super(FCOSLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.epsilon = epsilon
        self.strides = strides
        self.mi = mi
        self.use_center_sample = use_center_sample
        self.center_sample_radius = center_sample_radius

    def forward(self, cls_heads, reg_heads, center_heads, annotations):
        """
        compute cls loss, reg loss and center-ness loss in one batch
        """
        device = annotations.device
        batch_size = cls_heads[0].shape[0]
        image_h = cls_heads[0].shape[1] * self.strides[0]
        image_w = cls_heads[0].shape[2] * self.strides[0]
        fpn_feature_sizes = [[int(image_h/item), int(image_w/item)] for item in self.strides]
        fpn_feature_sizes = torch.tensor(fpn_feature_sizes, device=device)
        batch_positions = self.FCOSPositions(batch_size=batch_size, fpn_feature_sizes=fpn_feature_sizes)
        cls_preds, reg_preds, center_preds, batch_targets = self.get_batch_position_annotations(
            cls_heads, reg_heads, center_heads, batch_positions, annotations)

        cls_preds = torch.sigmoid(cls_preds)
        reg_preds = torch.exp(reg_preds)
        center_preds = torch.sigmoid(center_preds)

        reg_preds = reg_preds.type_as(cls_preds)
        center_preds = center_preds.type_as(cls_preds)
        batch_targets = batch_targets.type_as(cls_preds)

        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        reg_preds = reg_preds.view(-1, reg_preds.shape[-1])
        center_preds = center_preds.view(-1, center_preds.shape[-1])
        batch_targets = batch_targets.view(-1, batch_targets.shape[-1])


        cls_loss = self.compute_batch_focal_loss(cls_preds, batch_targets)
        reg_loss = self.compute_batch_giou_loss(reg_preds, batch_targets)
        center_ness_loss = self.compute_batch_centerness_loss(
            center_preds, batch_targets)

        return cls_loss, reg_loss, center_ness_loss

    def compute_batch_focal_loss(self, cls_preds, batch_targets):
        """
        compute batch focal loss(cls loss)
        cls_preds:[batch_size*points_num,num_classes]
        batch_targets:[batch_size*points_num,8]
        """
        cls_preds = torch.clamp(cls_preds,
                                min=self.epsilon,
                                max=1. - self.epsilon)
        positive_points_num = batch_targets[batch_targets[:, 4] > 0].shape[0]
        num_classes = cls_preds.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_targets[:, 4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(cls_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), cls_preds,
                         1. - cls_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        batch_bce_loss = -(
            loss_ground_truth * torch.log(cls_preds) +
            (1. - loss_ground_truth) * torch.log(1. - cls_preds))

        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / max(positive_points_num,1)

        return batch_focal_loss

    def compute_batch_giou_loss(self, reg_preds, batch_targets):
        """
        compute batch giou loss(reg loss)
        reg_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        """
        # only use positive points sample to compute reg loss
        reg_preds = reg_preds[batch_targets[:, 4] > 0]
        batch_targets = batch_targets[batch_targets[:, 4] > 0]
        positive_points_num = batch_targets.shape[0]


        center_ness_targets = batch_targets[:, 5]

        pred_bboxes_xy_min = batch_targets[:, 6:8] - reg_preds[:, 0:2]
        pred_bboxes_xy_max = batch_targets[:, 6:8] + reg_preds[:, 2:4]
        gt_bboxes_xy_min = batch_targets[:, 6:8] - batch_targets[:, 0:2]
        gt_bboxes_xy_max = batch_targets[:, 6:8] + batch_targets[:, 2:4]

        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                axis=1)
        gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], axis=1)

        overlap_area_top_left = torch.max(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        overlap_area_bot_right = torch.min(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                         overlap_area_top_left,
                                         min=0)
        overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

        # anchors and annotations convert format to [x1,y1,w,h]
        pred_bboxes_w_h = pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1
        gt_bboxes_w_h = gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2] + 1

        # compute anchors_area and annotations_area
        pred_bboxes_area = pred_bboxes_w_h[:, 0] * pred_bboxes_w_h[:, 1]
        gt_bboxes_area = gt_bboxes_w_h[:, 0] * gt_bboxes_w_h[:, 1]

        # compute union_area
        union_area = pred_bboxes_area + gt_bboxes_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        # compute ious between one image anchors and one image annotations
        ious = overlap_area / union_area

        enclose_area_top_left = torch.min(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        enclose_area_bot_right = torch.max(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                         enclose_area_top_left,
                                         min=0)
        enclose_area = enclose_area_sizes[:, 0] * enclose_area_sizes[:, 1]
        enclose_area = torch.clamp(enclose_area, min=1e-4)

        gious_loss = 1. - ious + (enclose_area - union_area) / enclose_area
        gious_loss = torch.clamp(gious_loss, min=0.0, max=2.0)
        # use center_ness_targets as the weight of gious loss
        gious_loss = gious_loss * center_ness_targets
        gious_loss = gious_loss.sum() / max(positive_points_num,1)
        gious_loss = self.reg_weight * gious_loss

        return gious_loss

    def compute_batch_centerness_loss(self, center_preds, batch_targets):
        """
        compute batch center_ness loss(center ness loss)
        center_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        """
        # only use positive points sample to compute center_ness loss
        center_preds = center_preds[batch_targets[:, 4] > 0]
        batch_targets = batch_targets[batch_targets[:, 4] > 0]
        positive_points_num = batch_targets.shape[0]


        center_ness_targets = batch_targets[:, 5:6]

        center_ness_loss = -(
            center_ness_targets * torch.log(center_preds) +
            (1. - center_ness_targets) * torch.log(1. - center_preds))
        center_ness_loss = center_ness_loss.sum() / max(positive_points_num,1)

        return center_ness_loss

    def get_batch_position_annotations(self, cls_heads, reg_heads,
                                       center_heads, batch_positions,
                                       annotations):
        """
        Assign a ground truth target for each position on feature map
        """
        device = annotations.device
        batch_mi, batch_stride = [], []
        for reg_head, mi, stride in zip(reg_heads, self.mi, self.strides):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_mi = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = per_level_mi + mi
            batch_mi.append(per_level_mi)
            per_level_stride = torch.zeros(B, H, W, 1).to(device)
            per_level_stride = per_level_stride + stride
            batch_stride.append(per_level_stride)

        cls_preds,reg_preds,center_preds,all_points_position,all_points_mi,all_points_stride=[],[],[],[],[],[]
        for cls_pred, reg_pred, center_pred, per_level_position, per_level_mi, per_level_stride in zip(
                cls_heads, reg_heads, center_heads, batch_positions, batch_mi,
                batch_stride):
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            center_pred = center_pred.view(center_pred.shape[0], -1,
                                           center_pred.shape[-1])
            per_level_position = per_level_position.view(
                per_level_position.shape[0], -1, per_level_position.shape[-1])
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1,
                                             per_level_mi.shape[-1])
            per_level_stride = per_level_stride.view(
                per_level_stride.shape[0], -1, per_level_stride.shape[-1])

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            center_preds.append(center_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)
            all_points_stride.append(per_level_stride)

        cls_preds = torch.cat(cls_preds, axis=1)
        reg_preds = torch.cat(reg_preds, axis=1)
        center_preds = torch.cat(center_preds, axis=1)
        all_points_position = torch.cat(all_points_position, axis=1)
        all_points_mi = torch.cat(all_points_mi, axis=1)
        all_points_stride = torch.cat(all_points_stride, axis=1)

        batch_targets = []
        for per_image_position, per_image_mi, per_image_stride, per_image_annotations in zip(
                all_points_position, all_points_mi, all_points_stride,
                annotations):
            per_image_annotations = per_image_annotations[
                per_image_annotations[:, 4] >= 0]
            points_num = per_image_position.shape[0]

            if per_image_annotations.shape[0] == 0:
                # 6:l,t,r,b,class_index,center-ness_gt
                per_image_targets = torch.zeros([points_num, 6], device=device)
            else:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                candidates = torch.zeros([points_num, annotaion_num, 4],
                                         device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)

                per_image_position = per_image_position.unsqueeze(1).repeat(
                    1, annotaion_num, 1)

                if self.use_center_sample:
                    candidates_center = (candidates[:, :, 2:4] +
                                         candidates[:, :, 0:2]) / 2
                    judge_distance = per_image_stride * self.center_sample_radius
                    judge_distance = judge_distance.repeat(1, annotaion_num)

                candidates[:, :,
                           0:2] = per_image_position[:, :,
                                                     0:2] - candidates[:, :,
                                                                       0:2]
                candidates[:, :,
                           2:4] = candidates[:, :,
                                             2:4] - per_image_position[:, :,
                                                                       0:2]

                candidates_min_value, _ = candidates.min(axis=-1, keepdim=True)
                sample_flag = (candidates_min_value[:, :, 0] >
                               0).int().unsqueeze(-1)
                # get all negative reg targets which points ctr out of gt box
                candidates = candidates * sample_flag

                # if use center sample get all negative reg targets which points not in center circle
                if self.use_center_sample:
                    compute_distance = torch.sqrt(
                        (per_image_position[:, :, 0] -
                         candidates_center[:, :, 0])**2 +
                        (per_image_position[:, :, 1] -
                         candidates_center[:, :, 1])**2)
                    center_sample_flag = (compute_distance <
                                          judge_distance).int().unsqueeze(-1)
                    candidates = candidates * center_sample_flag

                # get all negative reg targets which assign ground turth not in range of mi
                candidates_max_value, _ = candidates.max(axis=-1, keepdim=True)
                per_image_mi = per_image_mi.unsqueeze(1).repeat(
                    1, annotaion_num, 1)
                m1_negative_flag = (candidates_max_value[:, :, 0] >
                                    per_image_mi[:, :, 0]).int().unsqueeze(-1)
                candidates = candidates * m1_negative_flag
                m2_negative_flag = (candidates_max_value[:, :, 0] <
                                    per_image_mi[:, :, 1]).int().unsqueeze(-1)
                candidates = candidates * m2_negative_flag

                final_sample_flag = candidates.sum(axis=-1).sum(axis=-1)
                final_sample_flag = final_sample_flag > 0
                positive_index = (final_sample_flag == True).nonzero().squeeze(
                    dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    device=device)
                else:
                    positive_candidates = candidates[positive_index]

                    del candidates

                    sample_box_gts = per_image_annotations[:, 0:4].unsqueeze(0)
                    sample_box_gts = sample_box_gts.repeat(
                        positive_candidates.shape[0], 1, 1)
                    sample_class_gts = per_image_annotations[:, 4].unsqueeze(
                        -1).unsqueeze(0)
                    sample_class_gts = sample_class_gts.repeat(
                        positive_candidates.shape[0], 1, 1)

                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    device=device)

                    if positive_candidates.shape[1] == 1:
                        # if only one candidate for each positive sample
                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        # class_index value from 1 to 80 represent 80 positive classes
                        # class_index value 0 represenet negative class
                        positive_candidates = positive_candidates.squeeze(1)
                        sample_class_gts = sample_class_gts.squeeze(1)
                        per_image_targets[positive_index,
                                          0:4] = positive_candidates
                        per_image_targets[positive_index,
                                          4:5] = sample_class_gts + 1

                        l, t, r, b = per_image_targets[
                            positive_index, 0:1], per_image_targets[
                                positive_index, 1:2], per_image_targets[
                                    positive_index,
                                    2:3], per_image_targets[positive_index,
                                                            3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))
                    else:
                        # if a positive point sample have serveral object candidates,then choose the smallest area object candidate as the ground turth for this positive point sample
                        gts_w_h = sample_box_gts[:, :,
                                                 2:4] - sample_box_gts[:, :,
                                                                       0:2]
                        gts_area = gts_w_h[:, :, 0] * gts_w_h[:, :, 1]
                        positive_candidates_value = positive_candidates.sum(
                            axis=2)

                        # make sure all negative candidates areas==100000000,thus .min() operation wouldn't choose negative candidates
                        INF = 100000000
                        inf_tensor = torch.ones_like(gts_area) * INF
                        gts_area = torch.where(
                            torch.eq(positive_candidates_value, 0.),
                            inf_tensor, gts_area)

                        # get the smallest object candidate index
                        _, min_index = gts_area.min(axis=1)
                        candidate_indexes = (
                            torch.linspace(1, positive_candidates.shape[0],
                                           positive_candidates.shape[0]) -
                            1).long()
                        final_candidate_reg_gts = positive_candidates[
                            candidate_indexes, min_index, :]
                        final_candidate_cls_gts = sample_class_gts[
                            candidate_indexes, min_index]

                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        per_image_targets[positive_index,
                                          0:4] = final_candidate_reg_gts
                        per_image_targets[positive_index,
                                          4:5] = final_candidate_cls_gts + 1

                        l, t, r, b = per_image_targets[
                            positive_index, 0:1], per_image_targets[
                                positive_index, 1:2], per_image_targets[
                                    positive_index,
                                    2:3], per_image_targets[positive_index,
                                                            3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))

            per_image_targets = per_image_targets.unsqueeze(0)
            batch_targets.append(per_image_targets)

        batch_targets = torch.cat(batch_targets, axis=0)
        batch_targets = torch.cat([batch_targets, all_points_position], axis=2)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        return cls_preds, reg_preds, center_preds, batch_targets
    
    def FCOSPositions(self, batch_size, fpn_feature_sizes):

        """
        generate batch positions
        """
        device = fpn_feature_sizes.device
        one_sample_positions = []
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            featrue_positions = torch.tensor(featrue_positions, device=device)
            one_sample_positions.append(featrue_positions)

        batch_positions = []
        for per_level_featrue_positions in one_sample_positions:
            per_level_featrue_positions = per_level_featrue_positions.unsqueeze(
                0).repeat(batch_size, 1, 1, 1)
            batch_positions.append(per_level_featrue_positions)

        # if input size:[B,3,640,640]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]
        # per position format:[x_center,y_center]
        return batch_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        """
        generate all positions on a feature map
        """

        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (torch.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (torch.arange(0, feature_map_size[1]) + 0.5) * stride

        # feature_map_positions shape:[w,h,2] -> [h,w,2] -> [h*w,2]
        feature_map_positions = torch.tensor(np.array([[[shift_x, shift_y]
                                            for shift_y in shifts_y]
                                            for shift_x in shifts_x
                                            ])).permute(1, 0, 2).contiguous()

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions


import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetLoss(nn.Module):
    def __init__(self,
                 alpha=2.,
                 beta=4.,
                 wh_weight=0.1,
                 epsilon=1e-4,
                 min_overlap=0.7,
                 max_object_num=100):
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.wh_weight = wh_weight
        self.epsilon = epsilon
        self.min_overlap = min_overlap
        self.max_object_num = max_object_num

    def forward(self, heatmap_heads, offset_heads, wh_heads, annotations):
        """
        compute heatmap loss, offset loss and wh loss in one batch
        """
        batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask = self.get_batch_targets(
            heatmap_heads, annotations)

        heatmap_heads = torch.sigmoid(heatmap_heads)
        B, num_classes = heatmap_heads.shape[0], heatmap_heads.shape[1]
        heatmap_heads = heatmap_heads.permute(0, 2, 3, 1).contiguous().view(
            B, -1, num_classes)
        batch_heatmap_targets = batch_heatmap_targets.permute(
            0, 2, 3, 1).contiguous().view(B, -1, num_classes)

        wh_heads = wh_heads.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        offset_heads = offset_heads.permute(0, 2, 3,
                                            1).contiguous().view(B, -1, 2)

        wh_heads = wh_heads.type_as(heatmap_heads)
        offset_heads = offset_heads.type_as(heatmap_heads)
        batch_heatmap_targets = batch_heatmap_targets.type_as(heatmap_heads)
        batch_wh_targets = batch_wh_targets.type_as(heatmap_heads)
        batch_offset_targets = batch_offset_targets.type_as(heatmap_heads)
        batch_reg_to_heatmap_index = batch_reg_to_heatmap_index.type_as(
            heatmap_heads)
        batch_positive_targets_mask = batch_positive_targets_mask.type_as(
            heatmap_heads)

        heatmap_loss, offset_loss, wh_loss = [], [], []
        valid_image_num = 0
        device = annotations.device
        for per_heatmap_heads, per_wh_heads, per_offset_heads, per_heatmap_targets, per_wh_targets, per_offset_targets, per_reg_to_heatmap_index, per_positive_targets_mask in zip(
                heatmap_heads, wh_heads, offset_heads, batch_heatmap_targets,
                batch_wh_targets, batch_offset_targets,
                batch_reg_to_heatmap_index, batch_positive_targets_mask):
            # if no centers on heatmap_targets,this image is not valid
            # valid_center_num = (
            #     per_heatmap_targets[per_heatmap_targets == 1.]).shape[0]

            # if valid_center_num == 0:
            #     heatmap_loss.append(torch.tensor(0.).to(device))
            #     offset_loss.append(torch.tensor(0.).to(device))
            #     wh_loss.append(torch.tensor(0.).to(device))
            # else:
            valid_image_num += 1
            one_image_focal_loss = self.compute_one_image_focal_loss(
                per_heatmap_heads, per_heatmap_targets)
            one_image_offsetl1_loss = self.compute_one_image_offsetl1_loss(
                per_offset_heads, per_offset_targets,
                per_reg_to_heatmap_index, per_positive_targets_mask)
            one_image_whl1_loss = self.compute_one_image_whl1_loss(
                per_wh_heads, per_wh_targets, per_reg_to_heatmap_index,
                per_positive_targets_mask)

            heatmap_loss.append(one_image_focal_loss)
            offset_loss.append(one_image_offsetl1_loss)
            wh_loss.append(one_image_whl1_loss)

        if valid_image_num == 0:
            heatmap_loss = sum(heatmap_loss)
            offset_loss = sum(offset_loss)
            wh_loss = sum(wh_loss)
        else:
            heatmap_loss = sum(heatmap_loss) / valid_image_num
            offset_loss = sum(offset_loss) / valid_image_num
            wh_loss = sum(wh_loss) / valid_image_num

        return heatmap_loss, offset_loss, wh_loss

    def compute_one_image_focal_loss(self, per_image_heatmap_heads,
                                     per_image_heatmap_targets):
        device = per_image_heatmap_heads.device
        per_image_heatmap_heads = torch.clamp(per_image_heatmap_heads,
                                              min=self.epsilon,
                                              max=1. - self.epsilon)
        valid_center_num = (per_image_heatmap_targets[per_image_heatmap_targets
                                                      == 1.]).shape[0]

        # all center points
        positive_indexes = (per_image_heatmap_targets == 1.)
        # all non center points
        negative_indexes = (per_image_heatmap_targets < 1.)

        positive_loss = torch.log(per_image_heatmap_heads) * torch.pow(
            1 - per_image_heatmap_heads, self.alpha) * positive_indexes
        negative_loss = torch.log(1 - per_image_heatmap_heads) * torch.pow(
            per_image_heatmap_heads, self.alpha) * torch.pow(
                1 - per_image_heatmap_targets, self.beta) * negative_indexes

        if valid_center_num == 0:
            loss = -negative_loss.sum()
        else:
            loss = -(positive_loss.sum() + negative_loss.sum()) / valid_center_num

        return loss

    def compute_one_image_offsetl1_loss(self,
                                        per_image_offset_heads,
                                        per_image_offset_targets,
                                        per_image_reg_to_heatmap_index,
                                        per_image_positive_targets_mask,
                                        factor=1.0 / 9.0):
        device = per_image_offset_heads.device
        per_image_reg_to_heatmap_index = per_image_reg_to_heatmap_index.unsqueeze(
            -1).repeat(1, 2)
        per_image_offset_heads = torch.gather(
            per_image_offset_heads, 0, per_image_reg_to_heatmap_index.long())

        valid_object_num = (per_image_positive_targets_mask[
            per_image_positive_targets_mask == 1.]).shape[0]


        per_image_positive_targets_mask = per_image_positive_targets_mask.unsqueeze(
            -1).repeat(1, 2)
        per_image_offset_heads = per_image_offset_heads * per_image_positive_targets_mask
        per_image_offset_targets = per_image_offset_targets * per_image_positive_targets_mask

        #smooth l1
        x = torch.abs(per_image_offset_heads - per_image_offset_targets)
        loss = torch.where(torch.ge(x, factor), x - 0.5 * factor,
                           0.5 * (x**2) / factor)
        loss = loss.sum() / (valid_object_num + 1e-4)

        return loss

    def compute_one_image_whl1_loss(self,
                                    per_image_wh_heads,
                                    per_image_wh_targets,
                                    per_image_reg_to_heatmap_index,
                                    per_image_positive_targets_mask,
                                    factor=1.0 / 9.0):
        device = per_image_wh_heads.device
        per_image_reg_to_heatmap_index = per_image_reg_to_heatmap_index.unsqueeze(
            -1).repeat(1, 2)
        per_image_wh_heads = torch.gather(
            per_image_wh_heads, 0, per_image_reg_to_heatmap_index.long())

        valid_object_num = (per_image_positive_targets_mask[
            per_image_positive_targets_mask == 1.]).shape[0]


        # if valid_object_num == 0:
        #     return torch.tensor(0.).to(device)

        per_image_positive_targets_mask = per_image_positive_targets_mask.unsqueeze(
            -1).repeat(1, 2)
        per_image_wh_heads = per_image_wh_heads * per_image_positive_targets_mask
        per_image_wh_targets = per_image_wh_targets * per_image_positive_targets_mask

        x = torch.abs(per_image_wh_heads - per_image_wh_targets)
        loss = torch.where(torch.ge(x, factor), x - 0.5 * factor,
                           0.5 * (x**2) / factor)
        loss = loss.sum() / (valid_object_num + 1e-4)
        loss = self.wh_weight * loss

        return loss

    def get_batch_targets(self, heatmap_heads, annotations):
        B, num_classes, H, W = heatmap_heads.shape[0], heatmap_heads.shape[
            1], heatmap_heads.shape[2], heatmap_heads.shape[3]
        device = annotations.device

        batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask=[],[],[],[],[]
        for per_image_annots in annotations:
            # limit max annots num for per image
            per_image_annots = per_image_annots[per_image_annots[:, 4] >= 0]
            # limit max object num
            num_objs = min(per_image_annots.shape[0], self.max_object_num)

            per_image_heatmap_targets = torch.zeros((num_classes, H, W),
                                                    device=device)
            per_image_wh_targets = torch.zeros((self.max_object_num, 2),
                                               device=device)
            per_image_offset_targets = torch.zeros((self.max_object_num, 2),
                                                   device=device)
            per_image_positive_targets_mask = torch.zeros(
                (self.max_object_num, ), device=device)
            per_image_reg_to_heatmap_index = torch.zeros(
                (self.max_object_num, ), device=device)
            gt_bboxes, gt_classes = per_image_annots[:,
                                                     0:4], per_image_annots[:,
                                                                            4]
            # gt_bboxes divided by 4 to get downsample bboxes
            gt_bboxes = gt_bboxes / 4
            gt_bboxes[:,[0,2]] = torch.clamp(gt_bboxes[:,[0,2]], min=0, max=W-1)
            gt_bboxes[:,[1,3]] = torch.clamp(gt_bboxes[:,[1,3]], min=0, max=H-1)
            # make sure all height and width >0
            all_h, all_w = gt_bboxes[:,
                                     3] - gt_bboxes[:,
                                                    1], gt_bboxes[:,
                                                                  2] - gt_bboxes[:,
                                                                                 0]

            per_image_wh_targets[0:num_objs, 0] = all_w
            per_image_wh_targets[0:num_objs, 1] = all_h

            centers = torch.cat(
                [((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2).unsqueeze(-1),
                 ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2).unsqueeze(-1)],
                axis=1)
            centers_int = torch.trunc(centers)
            centers_decimal = torch.frac(centers)

            per_image_offset_targets[0:num_objs, :] = centers_decimal
            per_image_positive_targets_mask[0:num_objs] = 1

            per_image_reg_to_heatmap_index[
                0:num_objs] = centers_int[:, 1] * W + centers_int[:, 0]

            all_radius = self.compute_objects_gaussian_radius((all_h, all_w))
            per_image_heatmap_targets = self.draw_umich_gaussian(
                per_image_heatmap_targets, gt_classes, centers_int, all_radius)

            batch_heatmap_targets.append(
                per_image_heatmap_targets.unsqueeze(0))
            batch_wh_targets.append(per_image_wh_targets.unsqueeze(0))
            batch_reg_to_heatmap_index.append(
                per_image_reg_to_heatmap_index.unsqueeze(0))
            batch_offset_targets.append(per_image_offset_targets.unsqueeze(0))
            batch_positive_targets_mask.append(
                per_image_positive_targets_mask.unsqueeze(0))

        batch_heatmap_targets = torch.cat(batch_heatmap_targets, axis=0)
        batch_wh_targets = torch.cat(batch_wh_targets, axis=0)
        batch_offset_targets = torch.cat(batch_offset_targets, axis=0)
        batch_reg_to_heatmap_index = torch.cat(batch_reg_to_heatmap_index,
                                               axis=0)
        batch_positive_targets_mask = torch.cat(batch_positive_targets_mask,
                                                axis=0)

        return batch_heatmap_targets, batch_wh_targets, batch_offset_targets, batch_reg_to_heatmap_index, batch_positive_targets_mask

    def compute_objects_gaussian_radius(self, objects_size):
        all_h, all_w = objects_size
        all_h, all_w = torch.ceil(all_h), torch.ceil(all_w)

        a1 = 1
        b1 = (all_h + all_w)
        c1 = all_w * all_h * (1 - self.min_overlap) / (1 + self.min_overlap)
        sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (all_h + all_w)
        c2 = (1 - self.min_overlap) * all_w * all_h
        sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * self.min_overlap
        b3 = -2 * self.min_overlap * (all_h + all_w)
        c3 = (self.min_overlap - 1) * all_w * all_h
        sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        radius = torch.min(r1, r2)
        radius = torch.min(radius, r3)
        radius = torch.max(torch.zeros_like(radius), torch.trunc(radius))

        return radius

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def draw_umich_gaussian(self,
                            per_image_heatmap_targets,
                            gt_classes,
                            all_centers,
                            all_radius,
                            k=1):
        height, width = per_image_heatmap_targets.shape[
            1], per_image_heatmap_targets.shape[2]
        device = per_image_heatmap_targets.device

        for per_class, per_center, per_radius in zip(gt_classes, all_centers,
                                                     all_radius):
            per_diameter = 2 * per_radius + 1
            per_diameter = int(per_diameter.item())
            gaussian = self.gaussian2D((per_diameter, per_diameter),
                                       sigma=per_diameter / 6)
            gaussian = torch.FloatTensor(gaussian).to(device)

            x, y = per_center[0], per_center[1]
            left, right = min(x, per_radius), min(width - x, per_radius + 1)
            top, bottom = min(y, per_radius), min(height - y, per_radius + 1)

            masked_heatmap = per_image_heatmap_targets[per_class.long(), (
                y - top).long():(y +
                                 bottom).long(), (x -
                                                  left).long():(x +
                                                                right).long()]
            masked_gaussian = gaussian[(per_radius -
                                        top).long():(per_radius +
                                                     bottom).long(),
                                       (per_radius -
                                        left).long():(per_radius +
                                                      right).long()]

            if min(masked_gaussian.shape) > 0 and min(
                    masked_heatmap.shape) > 0:
                # 如果高斯图重叠，重叠点取最大值
                masked_heatmap = torch.max(masked_heatmap, masked_gaussian * k)

            per_image_heatmap_targets[per_class.long(),
                                      (y - top).long():(y + bottom).long(),
                                      (x - left).long():(
                                          x + right).long()] = masked_heatmap

        return per_image_heatmap_targets

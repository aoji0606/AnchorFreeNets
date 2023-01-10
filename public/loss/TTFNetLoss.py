import numpy as np

import torch
import torch.nn as nn


def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    return areas[:, None] if keep_axis else areas


class TTFNetLoss(nn.Module):
    def __init__(self,
                 focal_alpha=2.,
                 focal_beta=4.,
                 gaussian_radius_alpha=0.54,
                 hm_weight=1,
                 wh_weight=0.1,
                 epsilon=1e-4,
                 min_overlap=0.7,
                 max_object_num=100):
        super(TTFNetLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.gaussian_radius_alpha = gaussian_radius_alpha
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.epsilon = epsilon
        self.min_overlap = min_overlap
        self.max_object_num = max_object_num

        self.down_ratio = 4
        self.base_loc = None

    def forward(self, heatmap_heads, wh_heads, annotations):
        """
        compute heatmap loss and wh loss in one batch
        """
        ################
        # generate targets
        ################
        with torch.no_grad():
            batch_heatmap_targets, batch_box_targets, batch_reg_weights = self.get_batch_targets(
                heatmap_heads, annotations
            )

        ################
        # calculate loss
        ################
        B, num_classes, H, W = heatmap_heads.shape

        # prepare preds
        heatmap_heads = heatmap_heads.sigmoid_().permute(0, 2, 3, 1).contiguous().view(B, -1, num_classes)
        wh_heads = wh_heads.permute(0, 2, 3, 1).contiguous().view(B, -1, 4).type_as(heatmap_heads)

        # prepare targets
        batch_heatmap_targets = batch_heatmap_targets.permute(0, 2, 3, 1).contiguous().view(B, -1, num_classes)
        batch_heatmap_targets = batch_heatmap_targets.type_as(heatmap_heads)

        batch_box_targets = batch_box_targets.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        batch_box_targets = batch_box_targets.type_as(heatmap_heads)

        batch_reg_weights = batch_reg_weights.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
        batch_reg_weights = batch_reg_weights.type_as(heatmap_heads)

        # calulate heatmap loss
        heatmap_loss = self.compute_focal_loss(
            heatmap_heads, batch_heatmap_targets, self.focal_alpha, self.focal_beta
        ) * self.hm_weight

        # calulate wh loss: there is image scale bbox
        if self.base_loc is None or H * W != self.base_loc.shape[0]:
            base_step = self.down_ratio
            shifts_x = torch.arange(
                0, (W - 1) * base_step + 1, base_step, dtype=torch.float32, device=heatmap_heads.device
            )
            shifts_y = torch.arange(
                0, (H - 1) * base_step + 1, base_step, dtype=torch.float32, device=heatmap_heads.device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0).permute(1, 2, 0).contiguous().view(-1, 2)  # (hw, 2)

        pred_boxes = torch.cat((
            self.base_loc - wh_heads[:, :, [0, 1]], self.base_loc + wh_heads[:, :, [2, 3]]
        ), dim=2)  # (b, hw, 4)

        boxes = batch_box_targets  # (b, hw, 4)
        mask = batch_reg_weights.view(-1, H * W)  # (b, hw)
        avg_factor = mask.sum() + 1e-4
        wh_loss = self.compute_giou_loss(
            pred_boxes, boxes, mask, avg_factor=avg_factor
        ) * self.wh_weight

        return heatmap_loss, wh_loss

    def compute_focal_loss(self, pred, target, alpha=2.0, beta=4.):
        """
        Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
        gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

        Args:
            pred: tensor, any shape.
            target: tensor, same as pred.
            gamma: gamma in focal loss.

        Returns:

        """
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, beta)  # reduce punishment
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        return (pos_loss + neg_loss) / num_pos if num_pos > 0 else neg_loss

    def compute_giou_loss(self, pred, target, weight, avg_factor=None):
        """GIoU loss.
        Computing the GIoU loss between a set of predicted bboxes and target bboxes.
        pred: [num, 4], target: [num, 4], weight: [num,], avg_factor: float
        """
        pos_mask = weight > 0
        weight = weight[pos_mask].float()
        if avg_factor is None:
            avg_factor = torch.sum(pos_mask).float().item() + 1e-6
        bboxes1 = pred[pos_mask].view(-1, 4)
        bboxes2 = target[pos_mask].view(-1, 4)

        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
        enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

        overlap = wh[:, 0] * wh[:, 1]
        ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (ap + ag - overlap)

        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
        u = ap + ag - overlap
        gious = ious - (enclose_area - u) / enclose_area
        iou_distances = 1 - gious
        return torch.sum(iou_distances * weight)[None] / avg_factor

    def get_batch_targets(self, heatmap_heads, annotations):
        B, num_classes, H, W = heatmap_heads.shape  # [:4]
        device = annotations.device

        batch_heatmap_targets, batch_box_targets, batch_reg_weights = list(), list(), list()
        for per_image_annots in annotations:
            per_image_annots = per_image_annots[per_image_annots[:, 4] >= 0]
            gt_bboxes, gt_classes = per_image_annots[:, 0:4], per_image_annots[:, 4]

            # sort bboxes by area
            boxes_areas_log = bbox_areas(gt_bboxes).log()  # cal log(box area)
            boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))
            boxes_area_topk_log[:] = 1.  # norm boxes
            gt_bboxes = gt_bboxes[boxes_ind]
            gt_classes = gt_classes[boxes_ind]

            centers = torch.cat([
                ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2).unsqueeze(-1),
                ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2).unsqueeze(-1)
            ], axis=1) / self.down_ratio
            centers_int = torch.trunc(centers)

            # gt_bboxes divided by 4 to get downsample bboxes
            feat_gt_bboxes = gt_bboxes / self.down_ratio
            feat_gt_bboxes[:, [0, 2]] = torch.clamp(feat_gt_bboxes[:, [0, 2]], min=0, max=W - 1)
            feat_gt_bboxes[:, [1, 3]] = torch.clamp(feat_gt_bboxes[:, [1, 3]], min=0, max=H - 1)
            # make sure all height and width > 0
            feat_h = feat_gt_bboxes[:, 3] - feat_gt_bboxes[:, 1]
            feat_w = feat_gt_bboxes[:, 2] - feat_gt_bboxes[:, 0]
            feat_radius = self.compute_objects_gaussian_radius((feat_h, feat_w))

            # init per image targets
            per_image_heatmap_targets = torch.zeros((num_classes, H, W), device=device)
            per_image_box_targets = torch.ones((4, H, W), device=device) * -1
            per_image_reg_weights = torch.zeros((1, H, W), device=device)

            for i, (per_class, per_box, per_center, per_radius) in enumerate(
                    zip(gt_classes, gt_bboxes, centers_int, feat_radius)
            ):
                cls_id = per_class.long()

                # generate heatmap target
                one_cls_heatmap = torch.zeros((H, W), device=device)
                self.draw_truncate_gaussian(one_cls_heatmap, per_center, per_radius)
                per_image_heatmap_targets[cls_id] = torch.max(
                    per_image_heatmap_targets[cls_id], one_cls_heatmap
                )

                # generate box target
                box_target_inds = one_cls_heatmap > 0
                per_image_box_targets[:, box_target_inds] = per_box[:, None]

                # generate reg weight
                local_heatmap = one_cls_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[i]
                per_image_reg_weights[0, box_target_inds] = local_heatmap / ct_div

            batch_heatmap_targets.append(per_image_heatmap_targets.unsqueeze(0))
            batch_box_targets.append(per_image_box_targets.unsqueeze(0))
            batch_reg_weights.append(per_image_reg_weights.unsqueeze(0))

        batch_heatmap_targets = torch.cat(batch_heatmap_targets, axis=0)
        batch_box_targets = torch.cat(batch_box_targets, axis=0)
        batch_reg_weights = torch.cat(batch_reg_weights, axis=0)

        return batch_heatmap_targets.detach(), batch_box_targets.detach(), batch_reg_weights.detach()

    def compute_objects_gaussian_radius(self, objects_size):
        all_h, all_w = objects_size

        # ttfnet radius
        h_radius_alpha = (all_h / 2. * self.gaussian_radius_alpha).int()
        w_radius_alpha = (all_w / 2. * self.gaussian_radius_alpha).int()
        radius = torch.stack((h_radius_alpha, w_radius_alpha), dim=-1)

        # # centernet radius
        # all_h, all_w = torch.ceil(all_h), torch.ceil(all_w)
        # a1 = 1
        # b1 = (all_h + all_w)
        # c1 = all_w * all_h * (1 - self.min_overlap) / (1 + self.min_overlap)
        # sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
        # r1 = (b1 + sq1) / 2

        # a2 = 4
        # b2 = 2 * (all_h + all_w)
        # c2 = (1 - self.min_overlap) * all_w * all_h
        # sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
        # r2 = (b2 + sq2) / 2

        # a3 = 4 * self.min_overlap
        # b3 = -2 * self.min_overlap * (all_h + all_w)
        # c3 = (self.min_overlap - 1) * all_w * all_h
        # sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
        # r3 = (b3 + sq3) / 2

        # radius = torch.min(r1, r2)
        # radius = torch.min(radius, r3)
        # radius = torch.max(torch.zeros_like(radius), torch.trunc(radius))
        # # repeat(radius), to adopt ttfnet
        # radius = radius.unsqueeze(-1).repeat(1, 2)

        return radius

    def gaussian_2d(self, shape, sigma=(1, 1)):  # for ttfnet
        sigma_x, sigma_y = sigma

        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def draw_truncate_gaussian(self, heatmap, per_center, per_radius, k=1):
        height, width = heatmap.shape
        device = heatmap.device

        x, y = per_center[0].item(), per_center[1].item()
        h_radius, w_radius = per_radius[0].item(), per_radius[1].item()

        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        gaussian = self.gaussian_2d((h, w), sigma=(w / 6, h / 6))
        gaussian = torch.FloatTensor(gaussian).to(device)

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[
                         int(y - top):int(y + bottom),
                         int(x - left):int(x + right)
                         ]
        masked_gaussian = gaussian[
                          int(h_radius - top):int(h_radius + bottom),
                          int(w_radius - left):int(w_radius + right)
                          ]

        # If the Gaussians overlap, the overlap point takes the maximum value
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            masked_heatmap = torch.max(masked_heatmap, masked_gaussian * k)

        heatmap[
        int(y - top):int(y + bottom),
        int(x - left):int(x + right)
        ] = masked_heatmap

        return heatmap

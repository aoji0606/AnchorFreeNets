# -*- coding=utf-8 -*-
#!/usr/bin/python

"""
Comparing Gaussian heatmap differences in centernet and ttfnet.
1. read image and annotation
2. draw bbox
3. gerate gt, to get heatmap
4. draw gaussian (2 function)
5. save image
"""

import os
import cv2
import json
import numpy as np

import torch

# TODO: load voc anno
def read_voc_anno():
    pass


def read_coco_anno(anno_filename, img_basename):
    # read anno file
    with open(anno_filename) as f:
        anno_data = json.load(f)
    
    # find image id
    image_id = None
    for i in anno_data['images']:
        if i['file_name'] == img_basename:
            image_id = i['id']
    if image_id is None:
        raise ValueError(f'Can not find image in anno file, image basename is {img_basename}.')
    
    # generate category list
    category_list = {i['id']: i['name'] for i in anno_data['categories']}
    
    # find all bbox
    bbox_list = []
    for i in anno_data['annotations']:
        if i['image_id'] == image_id:
            bbox_list.append({
                'category_id': i['category_id'],
                'category': category_list[i['category_id']],
                'bbox': [
                    i['bbox'][0], i['bbox'][1],
                    i['bbox'][0] + i['bbox'][2], i['bbox'][1] + i['bbox'][3]
                ],
            })

    return bbox_list


def draw_bbox(image, bboxes, color=(128, 128, 128), txt_color=(255, 255, 255)):
    image = image.copy()
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    for bbox in bboxes:
        label = bbox['category']; box = bbox['bbox']
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA
        )
    return image


def draw_umich_gaussian(
    heatmaps,
    gt_classes,
    all_centers,
    all_radius,
    k=1
):
    height, width = heatmaps.shape[1:3]
    device = heatmaps.device

    for per_class, per_center, per_radius in zip(gt_classes, all_centers, all_radius):
        per_diameter = 2 * per_radius + 1
        per_diameter = int(per_diameter.item())
        gaussian = gaussian2D((per_diameter, per_diameter), sigma=per_diameter / 6)
        gaussian = torch.FloatTensor(gaussian).to(device)

        x, y = per_center[0], per_center[1]
        left, right = min(x, per_radius), min(width - x, per_radius + 1)
        top, bottom = min(y, per_radius), min(height - y, per_radius + 1)

        masked_heatmap = heatmaps[
            per_class.long(),
            (y - top).long():(y + bottom).long(),
            (x - left).long():(x + right).long()
        ]
        masked_gaussian = gaussian[
            (per_radius - top).long():(per_radius + bottom).long(),
            (per_radius - left).long():(per_radius + right).long()
        ]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            # 如果高斯图重叠，重叠点取最大值
            masked_heatmap = torch.max(masked_heatmap, masked_gaussian * k)

        heatmaps[
            per_class.long(),
            (y - top).long():(y + bottom).long(),
            (x - left).long():(x + right).long()
        ] = masked_heatmap

    return heatmaps


def compute_objects_gaussian_radius(objects_size, min_overlap=0.7):
    all_h, all_w = objects_size
    all_h, all_w = torch.ceil(all_h), torch.ceil(all_w)

    a1 = 1
    b1 = (all_h + all_w)
    c1 = all_w * all_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (all_h + all_w)
    c2 = (1 - min_overlap) * all_w * all_h
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (all_h + all_w)
    c3 = (min_overlap - 1) * all_w * all_h
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    radius = torch.min(r1, r2)
    radius = torch.min(radius, r3)
    radius = torch.max(torch.zeros_like(radius), torch.trunc(radius))

    return radius


def gaussian2D(shape, sigma=1):  # for centernet
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def gaussian_2d(shape, sigma_x=1, sigma_y=1):  # for ttfnet
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def generate_centernet_heatmaps(annos, image_shape=(512,512)):  # image_shape=(h, w)
    num_classes, H, W = 20, 128, 128

    h_downsmaple = image_shape[0] / H
    w_downsmaple = image_shape[1] / W

    # init heatmaps
    heatmaps = torch.zeros((num_classes, H, W))
    # init classes and bboxes
    gt_classes = torch.tensor([i['category_id'] for i in annos])
    gt_bboxes = torch.tensor([i['bbox'] for i in annos]).float()
    
    # gt_bboxes divided to get downsample bboxes
    gt_bboxes[:, [0, 2]] = gt_bboxes[:, [0, 2]] / w_downsmaple
    gt_bboxes[:, [1, 3]] = gt_bboxes[:, [1, 3]] / h_downsmaple
    # gt_bboxes = gt_bboxes / dowmsmaple

    # make sure all height and width > 0
    gt_bboxes[:,[0,2]] = torch.clamp(gt_bboxes[:,[0,2]], min=0, max=W-1)
    gt_bboxes[:,[1,3]] = torch.clamp(gt_bboxes[:,[1,3]], min=0, max=H-1)
    all_h, all_w = gt_bboxes[:, 3] - gt_bboxes[:, 1], gt_bboxes[:, 2] - gt_bboxes[:, 0]

    # generate center points
    centers = torch.cat([
        ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2).unsqueeze(-1),
        ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2).unsqueeze(-1)
    ], axis=1)
    centers_int = torch.trunc(centers)

    # calculate gaussian radius
    all_radius = compute_objects_gaussian_radius((all_h, all_w))

    # draw gaussian
    heatmaps = draw_umich_gaussian(heatmaps, gt_classes, centers_int, all_radius)

    return heatmaps


def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    gaussian = gaussian_2d((h, w), sigma_x=w/6, sigma_y=h/6)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[
        h_radius - top:h_radius + bottom,
        w_radius - left:w_radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def generate_ttfnet_heatmaps(annos, image_shape=(512,512), alpha=0.54, beta=0.54):
    num_classes, H, W = 20, 128, 128

    h_downsmaple = image_shape[0] / H
    w_downsmaple = image_shape[1] / W

    # init classes and bboxes
    gt_classes = torch.tensor([i['category_id'] for i in annos])
    gt_boxes = torch.tensor([i['bbox'] for i in annos]).float()
    # init heatmaps
    heatmaps = torch.zeros((num_classes, H, W))
    fake_heatmap = gt_boxes.new_zeros((H, W))
    # box_target = gt_boxes.new_ones((4, H, W)) * -1
    # reg_weight = gt_boxes.new_zeros((4 // 4, H, W))

    if True: # wh_area_process == 'log':
        boxes_areas_log = bbox_areas(gt_boxes).log()
    # sort by bbox area
    boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))
    if True: # wh_area_process == 'norm'
        boxes_area_topk_log[:] = 1.
    
    gt_boxes = gt_boxes[boxes_ind]
    gt_classes = gt_classes[boxes_ind]

    # feat_gt_boxes = torch.zeros_like(gt_boxes)
    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] / w_downsmaple
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] / h_downsmaple

    # feat_gt_boxes = gt_boxes / self.down_ratio
    feat_gt_boxes = gt_boxes
    feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0, max=W - 1)
    feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0, max=H - 1)
    feat_hs, feat_ws = (
        feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
        feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0]
    )

    # we calc the center and ignore area based on the gt-boxes of the origin scale
    # no peak will fall between pixels
    ct_ints = (
        torch.stack([
            (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
            (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        ], dim=1) # / self.down_ratio
    ).to(torch.int)

    h_radiuses_alpha = (feat_hs / 2. * alpha).int()
    w_radiuses_alpha = (feat_ws / 2. * alpha).int()
    # if alpha != beta: # and wh_gaussian is True
    #     h_radiuses_beta = (feat_hs / 2. * beta).int()
    #     w_radiuses_beta = (feat_ws / 2. * beta).int()

    # larger boxes have lower priority than small boxes.
    for k in range(boxes_ind.shape[0]):
        cls_id = gt_classes[k]

        fake_heatmap = fake_heatmap.zero_()
        draw_truncate_gaussian(
            fake_heatmap, ct_ints[k],
            h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item()
        )
        heatmaps[cls_id] = torch.max(heatmaps[cls_id], fake_heatmap)

        # if True: # self.wh_gaussian:
        #     if alpha != beta:
        #         fake_heatmap = fake_heatmap.zero_()
        #         draw_truncate_gaussian(
        #             fake_heatmap, ct_ints[k],
        #             h_radiuses_beta[k].item(), w_radiuses_beta[k].item()
        #         )
        #     box_target_inds = fake_heatmap > 0
        # else:
        #     ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
        #     box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
        #     box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

        # if True: # self.wh_agnostic:
        #     box_target[:, box_target_inds] = gt_boxes[k][:, None]
        #     cls_id = 0
        # else:
        #     box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

        # if True: # self.wh_gaussian:
        #     local_heatmap = fake_heatmap[box_target_inds]
        #     ct_div = local_heatmap.sum()
        #     local_heatmap *= boxes_area_topk_log[k]
        #     reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
        # else:
        #     reg_weight[cls_id, box_target_inds] = \
        #         boxes_area_topk_log[k] / box_target_inds.sum().float()
    
    return heatmaps


def merge_images(images):
    length = len(images)


def main():
    image_filename = '/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/val07/000001.jpg'
    anno_filename = '/home/jovyan/data-vol-polefs-1/datasets/voc0712_cocoformat/annotations/instances_val07.json'

    image = cv2.imread(image_filename)
    annos = read_coco_anno(anno_filename, os.path.basename(image_filename))
    # annos format: [{'category': 'class', 'bbox': [xmin, ymin, xmax, ymax]}, ...]
    print(annos)

    image = draw_bbox(image, annos)

    # generate centernet heatmaps
    centernet_heatmap = generate_centernet_heatmaps(annos, image_shape=image.shape[:2]) # torch.tensor, shape: c, h, w
    # np.array, shape: 128, 128, 3
    centernet_heatmap = centernet_heatmap.sum(dim=0, keepdim=True).repeat(3, 1, 1).permute(1, 2, 0)
    centernet_heatmap = centernet_heatmap.cpu().detach().numpy() * 255  # from 0-1 to 0-255
    centernet_heatmap = cv2.resize(
        centernet_heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
    )

    # generate ttfnet heatmaps
    ttfnet_heatmap = generate_ttfnet_heatmaps(annos, image_shape=image.shape[:2])
    ttfnet_heatmap = ttfnet_heatmap.sum(dim=0, keepdim=True).repeat(3, 1, 1).permute(1, 2, 0)
    ttfnet_heatmap = ttfnet_heatmap.cpu().detach().numpy() * 255  # from 0-1 to 0-255
    ttfnet_heatmap = cv2.resize(
        ttfnet_heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
    )

    # merge centernet_heatmap with image
    center_merged_image = image * 0.2 + centernet_heatmap * 0.8
    center_merged_image = cv2.resize(center_merged_image, (512,512), interpolation=cv2.INTER_AREA)
    ttf_merged_image = image * 0.2 + ttfnet_heatmap * 0.8
    ttf_merged_image = cv2.resize(ttf_merged_image, (512,512), interpolation=cv2.INTER_AREA)
    
    merged_image = np.concatenate((center_merged_image, ttf_merged_image), axis=1)

    cv2.imwrite('center_test.jpg', merged_image)


if __name__ == '__main__':
    main()
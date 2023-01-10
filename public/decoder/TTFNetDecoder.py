import torch
import torch.nn as nn
import torch.nn.functional as F


class TTFNetDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 min_score_threshold=0.05,
                 max_detection_num=100,
                 topk=100,
                 stride=4):
        super(TTFNetDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.min_score_threshold = min_score_threshold
        self.max_detection_num = max_detection_num
        self.topk = topk
        self.stride = stride

    def forward(self, heatmap_heads, wh_heads):
        device = heatmap_heads.device
        heatmap_heads = torch.sigmoid(heatmap_heads)

        batch_scores, batch_classes, batch_pred_bboxes = [], [], []
        for per_image_heatmap_heads, per_image_wh_heads in zip(heatmap_heads, wh_heads):
#             import pdb
#             pdb.set_trace()
            
            # filter and keep points which value large than the surrounding 8 points
            per_image_heatmap_heads = self.nms(per_image_heatmap_heads)
            topk_score, topk_indexes, topk_classes, topk_ys, topk_xs = self.get_topk(
                per_image_heatmap_heads, K=self.topk
            )
            
            per_image_wh_heads = per_image_wh_heads.permute(1, 2, 0).contiguous().view(-1, 4)
            per_image_wh_heads = per_image_wh_heads.gather(0, topk_indexes.repeat(1, 4))

            # image scale bbox
            topk_bboxes = torch.cat([
                topk_xs * self.stride - per_image_wh_heads[:, [0]],
                topk_ys * self.stride - per_image_wh_heads[:, [1]],
                topk_xs * self.stride + per_image_wh_heads[:, [2]],
                topk_ys * self.stride + per_image_wh_heads[:, [3]]
            ], dim=1)

            topk_bboxes[:, 0] = torch.clamp(topk_bboxes[:, 0], min=0)
            topk_bboxes[:, 1] = torch.clamp(topk_bboxes[:, 1], min=0)
            topk_bboxes[:, 2] = torch.clamp(topk_bboxes[:, 2], max=self.image_w - 1)
            topk_bboxes[:, 3] = torch.clamp(topk_bboxes[:, 3], max=self.image_h - 1)

            one_image_scores = (-1) * torch.ones((self.max_detection_num, ), device=device)
            one_image_classes = (-1) * torch.ones((self.max_detection_num, ), device=device)
            one_image_pred_bboxes = (-1) * torch.ones((self.max_detection_num, 4), device=device)

            topk_classes = topk_classes[topk_score > self.min_score_threshold].float()
            topk_bboxes = topk_bboxes[topk_score > self.min_score_threshold].float()
            topk_score = topk_score[topk_score > self.min_score_threshold].float()

            final_detection_num = min(self.max_detection_num, topk_score.shape[0])
            one_image_scores[0:final_detection_num] = topk_score[0:final_detection_num]
            one_image_classes[0:final_detection_num] = topk_classes[0:final_detection_num]
            one_image_pred_bboxes[0:final_detection_num, :] = topk_bboxes[0:final_detection_num, :]

            batch_scores.append(one_image_scores.unsqueeze(0))
            batch_classes.append(one_image_classes.unsqueeze(0))
            batch_pred_bboxes.append(one_image_pred_bboxes.unsqueeze(0))

        batch_scores = torch.cat(batch_scores, axis=0)
        batch_classes = torch.cat(batch_classes, axis=0)
        batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)

        # batch_scores shape:[batch_size,topk]
        # batch_classes shape:[batch_size,topk]
        # batch_pred_bboxes shape[batch_size,topk,4]
        return batch_scores, batch_classes, batch_pred_bboxes

    def nms(self, per_image_heatmap_heads, kernel=3):
        per_image_heatmap_max = F.max_pool2d(
            per_image_heatmap_heads,
            kernel,
            stride=1,
            padding=(kernel - 1) // 2
        )
        keep = (per_image_heatmap_max == per_image_heatmap_heads).float()

        return per_image_heatmap_heads * keep

    def get_topk(self, per_image_heatmap_heads, K):
        num_classes = per_image_heatmap_heads.shape[0]
        H = per_image_heatmap_heads.shape[1]
        W = per_image_heatmap_heads.shape[2]

        per_image_heatmap_heads = per_image_heatmap_heads.view(num_classes, -1)
        # 先取每个类别的heatmap上前k个最大激活点
        topk_scores, topk_indexes = torch.topk(
            per_image_heatmap_heads.view(num_classes, -1), K, dim=-1
        )

        # 取余，计算topk项在feature map上的y和x index(位置)
        topk_indexes = topk_indexes % (H * W)
        topk_ys = (topk_indexes / W).int().float()
        topk_xs = (topk_indexes % W).int().float()

        # 在topk_scores中取前k个最大分数(所有类别混合在一起再取)
        topk_score, topk_score_indexes = torch.topk(
            topk_scores.view(-1), K, dim=-1
        )

        # 整除K得到预测的类编号，因为heatmap view前第一个维度是类别数
        topk_classes = (topk_score_indexes / K).int()
        topk_score_indexes = topk_score_indexes.unsqueeze(-1)
        topk_indexes = torch.gather(topk_indexes.view(-1, 1), 0, topk_score_indexes)
        topk_ys = torch.gather(topk_ys.view(-1, 1), 0, topk_score_indexes)
        topk_xs = torch.gather(topk_xs.view(-1, 1), 0, topk_score_indexes)

        return topk_score, topk_indexes, topk_classes, topk_ys, topk_xs

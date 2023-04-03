import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import ssim_loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, teacher_feats, student_feats):
        if isinstance(teacher_feats, torch.Tensor):
            teacher_feats, student_feats = (teacher_feats,), (student_feats,)

        loss = 0.0
        for teacher_feat, student_feat in zip(teacher_feats, student_feats):
            loss += ssim_loss(teacher_feat, student_feat, window_size=11)

        return loss


class AT(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am


class RKDLoss(nn.Module):
    '''
    Relational Knowledge Distillation
    https://arxiv.org/pdf/1904.05068.pdf
    '''

    def __init__(self, w_dist, w_angle):
        super(RKDLoss, self).__init__()

        self.w_dist = w_dist
        self.w_angle = w_angle

    def forward(self, feats_s, feats_t):
        loss = 0.0
        for feat_s, feat_t in zip(feats_s, feats_t):
            feat_s = feat_s.sum(dim=1)
            feat_t = feat_t.sum(dim=1)
            loss += self.w_dist * self.rkd_dist(feat_s, feat_t) + self.w_angle * self.rkd_angle(feat_s, feat_t)

        return loss

    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

        feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist

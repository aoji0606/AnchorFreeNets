import torch
import torch.nn as nn
from kornia.losses import ssim_loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, preds_S, preds_T):
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S,), (preds_T,)

        loss = 0.
        for pred_S, pred_T in zip(preds_S, preds_T):
            assert pred_S.shape == pred_T.shape
            loss += ssim_loss(pred_S, pred_T, window_size=11)

        return loss

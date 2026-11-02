import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(1) @ g.unsqueeze(0)


class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        window = _gaussian_window(window_size, sigma)
        self.register_buffer('window', window.unsqueeze(0).unsqueeze(0))
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, gt):
        pred = pred.float()
        gt = gt.float()
        C = pred.shape[1]
        window = self.window.expand(C, -1, -1, -1)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_gt = F.conv2d(gt, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_gt_sq = mu_gt ** 2
        mu_cross = mu_pred * mu_gt

        sigma_pred_sq = (F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_pred_sq).clamp(min=0)
        sigma_gt_sq = (F.conv2d(gt * gt, window, padding=pad, groups=C) - mu_gt_sq).clamp(min=0)
        sigma_cross = F.conv2d(pred * gt, window, padding=pad, groups=C) - mu_cross

        ssim_map = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
                   ((mu_pred_sq + mu_gt_sq + self.C1) * (sigma_pred_sq + sigma_gt_sq + self.C2))

        return 1.0 - ssim_map.mean()

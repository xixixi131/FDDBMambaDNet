import torch
import torch.nn.functional as F


def compute_psnr(pred, gt, max_val=1.0):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)


def _gaussian_window(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)


def compute_ssim(pred, gt, window_size=11, sigma=1.5):
    C = pred.shape[1]
    window = _gaussian_window(window_size, sigma, pred.device).expand(C, -1, -1, -1)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
    mu_gt = F.conv2d(gt, window, padding=pad, groups=C)

    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_cross = mu_pred * mu_gt

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, window, padding=pad, groups=C) - mu_gt_sq
    sigma_cross = F.conv2d(pred * gt, window, padding=pad, groups=C) - mu_cross

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))

    return ssim_map.mean()

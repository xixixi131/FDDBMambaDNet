import torch
import torch.nn as nn


class PhaseLoss(nn.Module):
    """L1 loss on FFT phase spectrum with circular wrap handling."""

    def forward(self, pred, gt):
        pred = pred.float()
        gt = gt.float()
        fft_pred = torch.fft.rfft2(pred)
        fft_gt = torch.fft.rfft2(gt)

        phase_pred = torch.angle(fft_pred)
        phase_gt = torch.angle(fft_gt)

        diff = phase_pred - phase_gt
        phase_diff = torch.atan2(torch.sin(diff), torch.cos(diff))

        return phase_diff.abs().mean()

import torch
import torch.nn as nn


class FreqReconstructionLoss(nn.Module):
    """Frequency domain reconstruction loss on log-amplitude and phase."""

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        pred = pred.float()
        gt = gt.float()
        fft_pred = torch.fft.rfft2(pred)
        fft_gt = torch.fft.rfft2(gt)

        amp_pred = torch.log1p(fft_pred.abs())
        amp_gt = torch.log1p(fft_gt.abs())
        amp_loss = (amp_pred - amp_gt).abs().mean()

        phase_pred = torch.angle(fft_pred)
        phase_gt = torch.angle(fft_gt)
        diff = phase_pred - phase_gt
        phase_loss = torch.atan2(torch.sin(diff), torch.cos(diff)).abs().mean()

        return self.alpha * amp_loss + self.beta * phase_loss

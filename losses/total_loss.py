import torch.nn as nn

from .ssim_loss import SSIMLoss
from .phase_loss import PhaseLoss
from .perceptual_loss import PerceptualLoss
from .freq_reconstruction_loss import FreqReconstructionLoss


class TotalLoss(nn.Module):
    """Weighted combination of L1, SSIM, Perceptual, Phase, FreqReconstruction losses."""

    def __init__(self, lambda_l1=1.0, lambda_ssim=1.0, lambda_perceptual=0.1,
                 lambda_phase=0.5, lambda_freq_recon=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_phase = lambda_phase
        self.lambda_freq_recon = lambda_freq_recon

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss()
        self.phase = PhaseLoss()
        self.freq_recon = FreqReconstructionLoss()

    def forward(self, pred, gt, hazy=None):
        losses = {}
        total = 0.0

        l1_val = self.l1(pred, gt)
        losses['l1'] = l1_val.item()
        total = total + self.lambda_l1 * l1_val

        ssim_val = self.ssim(pred, gt)
        losses['ssim'] = ssim_val.item()
        total = total + self.lambda_ssim * ssim_val

        perceptual_val = self.perceptual(pred, gt)
        losses['perceptual'] = perceptual_val.item()
        total = total + self.lambda_perceptual * perceptual_val

        phase_val = self.phase(pred, gt)
        losses['phase'] = phase_val.item()
        total = total + self.lambda_phase * phase_val

        freq_val = self.freq_recon(pred, gt)
        losses['freq_recon'] = freq_val.item()
        total = total + self.lambda_freq_recon * freq_val

        losses['total'] = total.item()
        return total, losses

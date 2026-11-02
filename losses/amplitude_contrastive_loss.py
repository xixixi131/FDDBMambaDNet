import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:16]))
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)


class AmplitudeContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss on FFT amplitude in VGG feature space."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.temperature = temperature

    def _get_amplitude_feat(self, x):
        feat = self.vgg(x).float()
        amp = torch.fft.rfft2(feat).abs()
        amp = torch.log1p(amp)
        amp = amp.flatten(1)
        return F.normalize(amp, dim=1)

    def forward(self, pred, gt, hazy):
        with torch.no_grad():
            anchor = self._get_amplitude_feat(gt)
            negative = self._get_amplitude_feat(hazy)

        positive = self._get_amplitude_feat(pred)

        sim_pos = (anchor * positive).sum(dim=1) / self.temperature
        sim_neg = (anchor * negative).sum(dim=1) / self.temperature

        logits = torch.stack([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

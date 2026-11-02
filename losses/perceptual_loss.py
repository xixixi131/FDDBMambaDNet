import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG-19 perceptual loss at multiple layers."""

    def __init__(self, layer_weights=None):
        super().__init__()
        if layer_weights is None:
            layer_weights = [1.0, 1.0, 1.0, 1.0]

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg[:4])),    # relu1_2
            nn.Sequential(*list(vgg[4:9])),   # relu2_2
            nn.Sequential(*list(vgg[9:18])),  # relu3_4
            nn.Sequential(*list(vgg[18:27])), # relu4_4
        ])
        for p in self.parameters():
            p.requires_grad = False

        self.layer_weights = layer_weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, gt):
        pred = pred.float()
        gt = gt.float()
        pred = (pred - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        loss = 0.0
        x_pred, x_gt = pred, gt
        for i, block in enumerate(self.blocks):
            x_pred = block(x_pred)
            x_gt = block(x_gt)
            loss += self.layer_weights[i] * (x_pred - x_gt).abs().mean()
        return loss

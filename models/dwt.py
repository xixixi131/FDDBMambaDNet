import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTForward(nn.Module):
    """2D Haar Discrete Wavelet Transform."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        B, C, H, W = x.shape
        filters = self.filters.repeat(C, 1, 1, 1)
        x = F.conv2d(x, filters, stride=2, groups=C)
        x = x.reshape(B, C, 4, H // 2, W // 2)
        ll = x[:, :, 0]
        lh = x[:, :, 1]
        hl = x[:, :, 2]
        hh = x[:, :, 3]
        return ll, lh, hl, hh


class DWTInverse(nn.Module):
    """2D Haar Inverse Discrete Wavelet Transform."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        x = torch.stack([ll, lh, hl, hh], dim=2)
        x = x.reshape(B * C, 4, H, W)
        filters = self.filters
        x = F.conv_transpose2d(x, filters, stride=2, groups=1)
        x = x.reshape(B, C, H * 2, W * 2)
        return x

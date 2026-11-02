import torch.nn as nn
from .gabor_conv import GaborConv2d
from .utils import LayerNorm2d


class GaborBlock(nn.Module):
    """Single Gabor processing block for high-frequency subbands."""

    def __init__(self, in_channels, out_channels, n_orientations=4, n_scales=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.gabor1 = GaborConv2d(out_channels, out_channels, kernel_size=5,
                                  n_orientations=n_orientations, n_scales=n_scales)
        self.norm1 = LayerNorm2d(out_channels)
        self.gabor2 = GaborConv2d(out_channels, out_channels, kernel_size=7,
                                  n_orientations=n_orientations, n_scales=min(n_scales, 2))
        self.norm2 = LayerNorm2d(out_channels)
        self.mix = nn.Conv2d(out_channels, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        residual = x
        x = self.act(self.norm1(self.gabor1(x)))
        x = self.act(self.norm2(self.gabor2(x)))
        x = self.mix(x)
        return x + residual


class MultiScaleGaborBlock(nn.Module):
    """Stack of GaborBlocks for deeper high-frequency processing."""

    def __init__(self, in_channels, out_channels, n_blocks=2, n_orientations=4, n_scales=3):
        super().__init__()
        blocks = [GaborBlock(in_channels, out_channels, n_orientations, n_scales)]
        for _ in range(n_blocks - 1):
            blocks.append(GaborBlock(out_channels, out_channels, n_orientations, n_scales))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

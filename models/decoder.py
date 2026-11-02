import torch
import torch.nn as nn
from .dwt import DWTInverse
from .utils import LayerNorm2d


class RefinementBlock(nn.Module):
    """Conv-based refinement with residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class DecoderBlock(nn.Module):
    """One level of the U-Net decoder: generate HF → IDWT → concat skip → refine."""

    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.hf_gen = nn.Conv2d(in_dim, in_dim * 3, 1)
        self.idwt = DWTInverse()
        self.reduce = nn.Conv2d(in_dim + skip_dim, out_dim, 1)
        self.refine = RefinementBlock(out_dim)

    def forward(self, x, skip):
        B, C, H, W = x.shape
        hf = self.hf_gen(x)
        lh, hl, hh = hf.chunk(3, dim=1)

        up = self.idwt(x, lh, hl, hh)
        cat_feat = torch.cat([up, skip], dim=1)
        out = self.reduce(cat_feat)
        return self.refine(out)


class FinalUpsample(nn.Module):
    """Final IDWT upsample + output projection."""

    def __init__(self, dim, out_channels=3):
        super().__init__()
        self.hf_gen = nn.Conv2d(dim, dim * 3, 1)
        self.idwt = DWTInverse()
        self.out_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        hf = self.hf_gen(x)
        lh, hl, hh = hf.chunk(3, dim=1)
        up = self.idwt(x, lh, hl, hh)
        return self.out_conv(up)


class Decoder(nn.Module):
    """Full U-Net decoder with N levels."""

    def __init__(self, base_channels=32, num_levels=4, out_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_levels - 2, -1, -1):
            in_dim = base_channels * (2 ** (i + 1))
            skip_dim = base_channels * (2 ** i)
            out_dim = base_channels * (2 ** i)
            self.blocks.append(DecoderBlock(in_dim, skip_dim, out_dim))

        self.final = FinalUpsample(base_channels, out_channels)

    def forward(self, bottleneck, skips):
        x = bottleneck
        for i, block in enumerate(self.blocks):
            skip = skips[-(i + 1)]
            x = block(x, skip)
        return self.final(x)

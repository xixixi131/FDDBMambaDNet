import torch
import torch.nn as nn
from .dwt import DWTForward
from .mamba_branch import MambaStage
from .gabor_branch import MultiScaleGaborBlock
from .cross_fusion import CrossFusionModule


class DualBranchEncoderBlock(nn.Module):
    """One level of the U-Net encoder: DWT → dual branch → cross fusion."""

    def __init__(self, dim, mamba_blocks=2, gabor_blocks=2, num_heads=4,
                 d_state=64, d_conv=4, expand=2, n_orientations=4, n_scales=3):
        super().__init__()
        self.dwt = DWTForward()
        self.mamba_stage = MambaStage(dim, n_blocks=mamba_blocks,
                                      d_state=d_state, d_conv=d_conv, expand=expand)
        self.gabor_stage = MultiScaleGaborBlock(dim * 3, dim, n_blocks=gabor_blocks,
                                                n_orientations=n_orientations, n_scales=n_scales)
        self.cross_fusion = CrossFusionModule(dim, num_heads)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        hf = torch.cat([lh, hl, hh], dim=1)

        feat_low = self.mamba_stage(ll)
        feat_high = self.gabor_stage(hf)

        fused = self.cross_fusion(feat_low, feat_high)
        return fused, fused


class Encoder(nn.Module):
    """Full U-Net encoder with stem + N levels."""

    def __init__(self, in_channels=3, base_channels=32, num_levels=4,
                 mamba_blocks_per_level=None, gabor_blocks_per_level=None,
                 num_heads=4, d_state=64, d_conv=4, expand=2,
                 n_orientations=4, n_scales=3):
        super().__init__()
        if mamba_blocks_per_level is None:
            mamba_blocks_per_level = [2, 2, 2, 4]
        if gabor_blocks_per_level is None:
            gabor_blocks_per_level = [2, 2, 2, 4]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
        )

        self.levels = nn.ModuleList()
        self.channel_expands = nn.ModuleList()

        for i in range(num_levels):
            dim = base_channels * (2 ** i)
            self.levels.append(
                DualBranchEncoderBlock(
                    dim=dim,
                    mamba_blocks=mamba_blocks_per_level[i],
                    gabor_blocks=gabor_blocks_per_level[i],
                    num_heads=num_heads,
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    n_orientations=n_orientations, n_scales=n_scales,
                )
            )
            if i < num_levels - 1:
                next_dim = base_channels * (2 ** (i + 1))
                self.channel_expands.append(
                    nn.Conv2d(dim, next_dim, 1)
                )

        self.num_levels = num_levels

    def forward(self, x):
        x = self.stem(x)
        skips = []

        for i, level in enumerate(self.levels):
            fused, skip = level(x)
            if i < self.num_levels - 1:
                skips.append(skip)
                x = self.channel_expands[i](fused)
            else:
                return fused, skips

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class DFFNet(nn.Module):
    """Dual-Frequency Fusion Dehazing Network (U-Net architecture)."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=32, num_levels=4,
                 mamba_blocks_per_level=None, gabor_blocks_per_level=None,
                 d_state=64, d_conv=4, expand=2, num_heads=4,
                 gabor_n_orientations=4, gabor_n_scales=3,
                 use_residual_learning=True):
        super().__init__()
        if mamba_blocks_per_level is None:
            mamba_blocks_per_level = [2, 2, 2, 4]
        if gabor_blocks_per_level is None:
            gabor_blocks_per_level = [2, 2, 2, 4]

        self.use_residual_learning = use_residual_learning

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            mamba_blocks_per_level=mamba_blocks_per_level,
            gabor_blocks_per_level=gabor_blocks_per_level,
            num_heads=num_heads,
            d_state=d_state, d_conv=d_conv, expand=expand,
            n_orientations=gabor_n_orientations,
            n_scales=gabor_n_scales,
        )

        self.decoder = Decoder(
            base_channels=base_channels,
            num_levels=num_levels,
            out_channels=out_channels,
        )

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        out = self.decoder(bottleneck, skips)
        if self.use_residual_learning:
            out = x + out
        if not self.training:
            out = out.clamp(0, 1)
        return out

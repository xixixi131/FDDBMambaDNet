import torch
import torch.nn as nn
from .utils import LayerNorm2d

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class SimpleSSSM(nn.Module):
    """Fallback SSM when mamba-ssm is unavailable (CPU/Windows debugging)."""

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_inner)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = x_branch.transpose(1, 2)

        x_branch = self.act(x_branch)
        z = self.act(z)
        x_branch = x_branch * z

        x_branch = self.norm(x_branch)
        return self.out_proj(x_branch)


def build_ssm(d_model, d_state=64, d_conv=4, expand=2):
    if HAS_MAMBA:
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return SimpleSSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class Mamba2Block(nn.Module):
    """Mamba2-based block with 4-directional scanning for 2D features."""

    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.ssm_fwd_row = build_ssm(dim, d_state, d_conv, expand)
        self.ssm_bwd_row = build_ssm(dim, d_state, d_conv, expand)
        self.ssm_fwd_col = build_ssm(dim, d_state, d_conv, expand)
        self.ssm_bwd_col = build_ssm(dim, d_state, d_conv, expand)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)

        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        y_fwd_row = self.ssm_fwd_row(x_flat)

        x_bwd = x_flat.flip(1)
        y_bwd_row = self.ssm_bwd_row(x_bwd).flip(1)

        x_col = x.permute(0, 3, 2, 1).reshape(B, H * W, C)
        y_fwd_col = self.ssm_fwd_col(x_col)
        y_fwd_col = y_fwd_col.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

        x_bwd_col = x_col.flip(1)
        y_bwd_col = self.ssm_bwd_col(x_bwd_col).flip(1)
        y_bwd_col = y_bwd_col.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

        y = (y_fwd_row + y_bwd_row + y_fwd_col + y_bwd_col) / 4.0
        y = self.proj(y)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return y + residual


class MambaStage(nn.Module):
    """Stack of Mamba2Blocks for one encoder level."""

    def __init__(self, dim, n_blocks=2, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            Mamba2Block(dim, d_state, d_conv, expand) for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

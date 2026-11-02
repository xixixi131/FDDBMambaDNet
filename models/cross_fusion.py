import torch
import torch.nn as nn
from .utils import LayerNorm2d, FFN


class ChannelCrossAttention(nn.Module):
    """Transposed cross-attention operating in channel dimension: O(C^2) complexity."""

    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        N = H * W

        q = self.q_proj(q_feat).reshape(B, self.num_heads, self.head_dim, N)
        k = self.k_proj(kv_feat).reshape(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(kv_feat).reshape(B, self.num_heads, self.head_dim, N)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        return self.out_proj(out)


class Stage1UnidirectionalFusion(nn.Module):
    """Low-freq guides high-freq via unidirectional cross-attention."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm_q = LayerNorm2d(dim)
        self.norm_kv = LayerNorm2d(dim)
        self.cross_attn = ChannelCrossAttention(dim, num_heads)
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = FFN(dim, expansion_ratio=4)

    def forward(self, feat_low, feat_high):
        q = self.norm_q(feat_high)
        kv = self.norm_kv(feat_low)
        feat_high = feat_high + self.cross_attn(q, kv)
        feat_high = feat_high + self.ffn(self.norm_ffn(feat_high))
        return feat_high


class Stage2BidirectionalFusion(nn.Module):
    """Bidirectional cross-attention for deep fusion."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm_low_q = LayerNorm2d(dim)
        self.norm_high_kv = LayerNorm2d(dim)
        self.cross_attn_l2h = ChannelCrossAttention(dim, num_heads)
        self.norm_ffn_low = LayerNorm2d(dim)
        self.ffn_low = FFN(dim, expansion_ratio=4)

        self.norm_high_q = LayerNorm2d(dim)
        self.norm_low_kv = LayerNorm2d(dim)
        self.cross_attn_h2l = ChannelCrossAttention(dim, num_heads)
        self.norm_ffn_high = LayerNorm2d(dim)
        self.ffn_high = FFN(dim, expansion_ratio=4)

    def forward(self, feat_low, feat_high):
        q_l = self.norm_low_q(feat_low)
        kv_h = self.norm_high_kv(feat_high)
        feat_low = feat_low + self.cross_attn_l2h(q_l, kv_h)
        feat_low = feat_low + self.ffn_low(self.norm_ffn_low(feat_low))

        q_h = self.norm_high_q(feat_high)
        kv_l = self.norm_low_kv(feat_low)
        feat_high = feat_high + self.cross_attn_h2l(q_h, kv_l)
        feat_high = feat_high + self.ffn_high(self.norm_ffn_high(feat_high))

        return feat_low, feat_high


class CrossFusionModule(nn.Module):
    """Two-stage cross-fusion: unidirectional guidance + bidirectional deep fusion."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.stage1 = Stage1UnidirectionalFusion(dim, num_heads)
        self.stage2 = Stage2BidirectionalFusion(dim, num_heads)
        self.fuse = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, feat_low, feat_high):
        feat_high_guided = self.stage1(feat_low, feat_high)
        fused_low, fused_high = self.stage2(feat_low, feat_high_guided)
        return self.fuse(torch.cat([fused_low, fused_high], dim=1))

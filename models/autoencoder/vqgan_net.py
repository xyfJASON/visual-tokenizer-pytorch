"""
Encoder and Decoder following `taming-transformers` (vqgan).

References:
  - https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
  - https://github.com/FoundationVision/LlamaGen/blob/main/tokenizer/tokenizer_image/vq_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor):
        return self.block(x) + self.shortcut(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 1, groups: int = 32):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(groups, dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = (dim // n_heads) ** -0.5

    def forward(self, x: Tensor):
        bs, C, H, W = x.shape
        normx = self.norm(x)
        q = self.q(normx).view(bs * self.n_heads, -1, H*W)
        k = self.k(normx).view(bs * self.n_heads, -1, H*W)
        v = self.v(normx).view(bs * self.n_heads, -1, H*W)
        q = q * self.scale
        attn = torch.bmm(q.permute(0, 2, 1), k).softmax(dim=-1)
        output = torch.bmm(v, attn.permute(0, 2, 1)).view(bs, -1, H, W)
        output = self.proj(output)
        return output + x


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            z_channels: int = 256,
            dim: int = 128,
            dim_mults: tuple[int] = (1, 1, 2, 2, 4),
            num_res_blocks: int = 2,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.n_stages = len(dim_mults)

        # first conv
        self.conv_in = nn.Conv2d(in_channels, dim, 3, 1, 1)
        cur_dim = dim

        # downsampling
        self.down_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            out_dim = dim * dim_mults[i]
            blocks = []
            for j in range(num_res_blocks):
                blocks.append(ResBlock(cur_dim, out_dim, dropout=dropout))
                cur_dim = out_dim
                if i == self.n_stages - 1:
                    blocks.append(SelfAttentionBlock(out_dim))
            if i < self.n_stages - 1:
                blocks.append(Downsample(out_dim))
            self.down_blocks.append(nn.Sequential(*blocks))

        # bottleneck
        self.bottleneck_block = nn.Sequential(
            ResBlock(cur_dim, cur_dim, dropout=dropout),
            SelfAttentionBlock(cur_dim),
            ResBlock(cur_dim, cur_dim, dropout=dropout),
        )

        # out
        self.out_block = nn.Sequential(
            nn.GroupNorm(32, cur_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(cur_dim, z_channels, 3, 1, 1),
        )

    @property
    def z_channels(self):
        return self.out_block[-1].out_channels

    @property
    def downsample_factor(self):
        return 2 ** (self.n_stages - 1)

    def forward(self, x: Tensor):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.bottleneck_block(x)
        x = self.out_block(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            z_channels: int = 256,
            dim: int = 128,
            dim_mults: tuple[int] = (1, 1, 2, 2, 4),
            num_res_blocks: int = 2,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.n_stages = len(dim_mults)

        # in
        self.in_block = nn.Conv2d(z_channels, dim * dim_mults[-1], 3, 1, 1)
        cur_dim = dim * dim_mults[-1]

        # bottleneck
        self.bottleneck_block = nn.Sequential(
            ResBlock(cur_dim, cur_dim, dropout=dropout),
            SelfAttentionBlock(cur_dim),
            ResBlock(cur_dim, cur_dim, dropout=dropout),
        )

        # upsampling
        self.up_blocks = nn.ModuleList([])
        for i in range(self.n_stages - 1, -1, -1):
            out_dim = dim * dim_mults[i]
            blocks = []
            for j in range(num_res_blocks + 1):
                blocks.append(ResBlock(cur_dim, out_dim, dropout=dropout))
                cur_dim = out_dim
                if i == self.n_stages - 1:
                    blocks.append(SelfAttentionBlock(out_dim))
            if i > 0:
                blocks.append(Upsample(out_dim))
            self.up_blocks.append(nn.Sequential(*blocks))

        # last conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, cur_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(cur_dim, out_channels, 3, 1, 1),
        )

    @property
    def z_channels(self):
        return self.in_block.in_channels

    @property
    def upsample_factor(self):
        return 2 ** (self.n_stages - 1)

    @property
    def last_layer(self):
        return self.conv_out[-1].weight

    def forward(self, x: Tensor):
        x = self.in_block(x)
        x = self.bottleneck_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x

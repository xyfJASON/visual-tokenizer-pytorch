"""
Encoder and Decoder following vit-vqgan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        B, L, D = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=None, is_causal=False,
            dropout_p=self.dropout if self.training else 0,
        )

        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.out_dropout(self.out(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            image_size: int = 256,
            patch_size: int = 8,
            embed_dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        self.embed_dim = embed_dim

        # image to patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size ** 2, embed_dim))

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ffn
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        # weight initialization
        self.apply(init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)

    @property
    def z_channels(self):
        return self.embed_dim

    def forward(self, x: Tensor):
        # patch embedding
        x = self.patch_embed(x)                     # (B, D, H, W)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, H*W, D)
        x = x + self.pos_embed                      # (B, H*W, D)

        # transformer blocks
        for block in self.blocks:
            x = block(x)                            # (B, H*W, D)

        # ffn
        x = self.ffn(x)                             # (B, H*W, D)

        # reshape to 2D image
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, D, H, W)
        return x.contiguous()


class Decoder(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            image_size: int = 256,
            patch_size: int = 8,
            embed_dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        grid_size = image_size // patch_size
        self.embed_dim = embed_dim

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size ** 2, embed_dim))

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ffn
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

        # output projection
        self.ln = nn.LayerNorm(embed_dim)
        self.out = nn.Sequential(
            nn.Conv2d(embed_dim, patch_size * patch_size * out_channels, 1),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )

        # weight initialization
        self.apply(init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)

    @property
    def z_channels(self):
        return self.embed_dim

    def forward(self, x: Tensor):
        B, D, H, W = x.shape

        # reshape to 1D sequence
        x = x.flatten(2).transpose(1, 2)            # (B, H*W, D)
        x = x + self.pos_embed                      # (B, H*W, D)

        # transformer blocks
        for block in self.blocks:
            x = block(x)                            # (B, H*W, D)

        # ffn
        x = self.ffn(x)                             # (B, H*W, D)

        # output projection
        x = self.ln(x)                              # (B, H*W, D)
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, D, H, W)
        x = self.out(x.contiguous())                # (B, C, H, W)
        return x

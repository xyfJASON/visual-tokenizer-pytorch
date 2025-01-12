"""
Encoder and Decoder following titok (an image is worth 32 tokens).
We remove some unnecessary modules compared to the original implementation.

References:
  - https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/blocks.py
  - https://github.com/bytedance/1d-tokenizer/blob/main/modeling/titok.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
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

        q, k, v = self.qkv(x).chunk(3, dim=-1)                                  # (B, L, D)
        q = q.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)    # (B, H, L, D/H)
        k = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)    # (B, H, L, D/H)
        v = v.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)    # (B, H, L, D/H)

        x = F.scaled_dot_product_attention(                                     # (B, H, L, D/H)
            query=q, key=k, value=v, attn_mask=None, is_causal=False,
            dropout_p=self.dropout if self.training else 0,
        )

        x = x.transpose(1, 2).reshape(B, L, D)                                  # (B, L, D)
        x = self.out_dropout(self.out(x))                                       # (B, L, D)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
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
            patch_size: int = 16,
            embed_dim: int = 512,
            n_heads: int = 8,
            n_layers: int = 8,
            n_tokens: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.patch_size = patch_size
        self.n_tokens = n_tokens

        # patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.patch_pos_embed = nn.Parameter(torch.zeros(self.grid_size ** 2, embed_dim))

        # latent token embedding
        self.latent_token_embed = nn.Parameter(torch.zeros(n_tokens, embed_dim))

        # transformer blocks
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)])
        self.ln_post = nn.LayerNorm(embed_dim)

        # weights initialization
        self.apply(init_weights)
        nn.init.normal_(self.patch_pos_embed, std=0.02)
        nn.init.normal_(self.latent_token_embed, std=0.02)

    @property
    def z_channels(self):
        return self.embed_dim

    def forward(self, x: Tensor):
        B = x.shape[0]

        # patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1)                                               # (B, L, D)
        x = x + self.patch_pos_embed.unsqueeze(0)                                       # (B, L, D)

        # latent token embedding
        latent_tokens = self.latent_token_embed.unsqueeze(0).expand(B, -1, -1)          # (B, K, D)

        # concatentate
        x = torch.cat([x, latent_tokens], dim=1)                                        # (B, L+K, D)

        # transformer blocks
        x = self.ln_pre(x)
        for block in self.blocks:
            x = block(x)                                                                # (B, L+K, D)

        # output
        latent_tokens = x[:, -self.n_tokens:]                                           # (B, K, D)
        latent_tokens = self.ln_post(latent_tokens)                                     # (B, K, D)

        # reshape to fake 2D image (to be compatible with the quantizer)
        latent_tokens = latent_tokens.permute(0, 2, 1).unsqueeze(2)                     # (B, D, 1, K)
        return latent_tokens.contiguous()


class Decoder(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            image_size: int = 256,
            patch_size: int = 16,
            embed_dim: int = 512,
            n_heads: int = 8,
            n_layers: int = 8,
            n_tokens: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.patch_size = patch_size
        self.n_tokens = n_tokens

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(n_tokens, embed_dim))

        # mask token embedding
        self.mask_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.mask_pos_embed = nn.Parameter(torch.zeros(self.grid_size ** 2, embed_dim))

        # transformer blocks
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)])
        self.ln_post = nn.LayerNorm(embed_dim)

        # output projection
        self.out = nn.Sequential(
            nn.Conv2d(embed_dim, patch_size * patch_size * out_channels, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )

        # weights initialization
        self.apply(init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_embed, std=0.02)
        nn.init.normal_(self.mask_pos_embed, std=0.02)

    @property
    def z_channels(self):
        return self.embed_dim

    def forward(self, x: Tensor):
        # reshape to 1D tokens
        B, D, H, W = x.shape
        assert H == 1 and W == self.n_tokens
        x = x.reshape(B, D, W).permute(0, 2, 1)                                         # (B, K, D)
        x = x + self.pos_embed.unsqueeze(0)                                             # (B, K, D)

        # mask token embedding
        mask_tokens = self.mask_embed.unsqueeze(0).repeat(B, self.grid_size ** 2, 1)    # (B, L, D)
        mask_tokens = mask_tokens + self.mask_pos_embed.unsqueeze(0)                    # (B, L, D)

        # concatentate
        x = torch.cat([mask_tokens, x], dim=1)                                          # (B, L+K, D)

        # transformer blocks
        x = self.ln_pre(x)
        for block in self.blocks:
            x = block(x)                                                                # (B, L+K, D)

        # output
        x = x[:, :self.grid_size ** 2]                                                  # (B, L, D)
        x = self.ln_post(x)                                                             # (B, L, D)
        x = x.permute(0, 2, 1).reshape(B, -1, self.grid_size, self.grid_size)           # (B, D, H, W)
        x = self.out(x.contiguous())
        return x

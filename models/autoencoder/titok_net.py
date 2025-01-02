"""
Encoder and Decoder following titok (an image is worth 32 tokens).

References:
  - https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/blocks.py
"""

import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ResidualAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def attention(self, x: Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: Tensor):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            dim: int = 512,
            n_heads: int = 8,
            n_layers: int = 8,
            n_tokens: int = 32,
            image_size: int = 256,
            patch_size: int = 16,
    ):
        super().__init__()

        scale = dim ** 0.5
        self.dim = dim
        self.grid_size = image_size // patch_size
        self.patch_size = patch_size
        self.n_tokens = n_tokens

        # patch embedding and class embedding
        self.patch_emb = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.class_emb = nn.Parameter(torch.zeros(1, dim))
        self.patch_pos_emb = nn.Parameter(torch.zeros(self.grid_size ** 2 + 1, dim))

        # latent token embedding
        self.latent_token_emb = nn.Parameter(torch.zeros(n_tokens, dim))
        self.latent_token_pos_emb = nn.Parameter(torch.zeros(n_tokens, dim))

        # transformer blocks
        self.ln_pre = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            ResidualAttention(dim, n_heads)
            for _ in range(n_layers)
        ])
        self.ln_post = nn.LayerNorm(dim)

        # weights initialization
        self.apply(init_weights)
        nn.init.trun_normal_(self.class_emb, std=scale)  # TODO
        nn.init.trunc_normal_(self.patch_pos_emb, std=scale)
        nn.init.trunc_normal_(self.latent_token_emb, std=scale)
        nn.init.trunc_normal_(self.latent_token_pos_emb, std=scale)

    @property
    def z_channels(self):
        return self.dim

    def forward(self, x: Tensor):
        # patch embedding
        x = self.patch_emb(x)
        B, D, H, W = x.shape
        x = x.reshape(B, D, H * W).permute(0, 2, 1)                                     # (B, L, D)
        # prepend class embedding and add position embedding
        x = torch.cat([self.class_emb.unsqueeze(0).expand(B, -1, -1), x], dim=1)        # (B, L+1, D)
        x = x + self.patch_pos_emb.unsqueeze(0)                                         # (B, L+1, D)
        # latent token embedding
        latent_tokens = self.latent_token_emb.unsqueeze(0).expand(B, -1, -1)            # (B, K, D)
        latent_tokens = latent_tokens + self.latent_token_pos_emb.unsqueeze(0)          # (B, K, D)
        # concatentate
        x = torch.cat([x, latent_tokens], dim=1)                                        # (B, L+K+1, D)
        # transformer blocks (note nn.MultiheadAttention uses `batch_first=False` by default)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in self.blocks:
            x = block(x)
        x = x.permute(1, 0, 2)                                                          # (B, L+K+1, D)
        # output
        latent_tokens = x[:, -self.n_tokens:]                                           # (B, K, D)
        latent_tokens = self.ln_post(latent_tokens)                                     # (B, K, D)
        # reshape to fake 2D image (to be compatible with the quantizer)
        latent_tokens = latent_tokens.permute(0, 2, 1).reshape(B, D, 1, self.n_tokens)  # (B, D', 1, K)
        return latent_tokens


class Decoder(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            dim: int = 512,
            n_heads: int = 8,
            n_layers: int = 8,
            n_tokens: int = 32,
            image_size: int = 256,
            patch_size: int = 16,
    ):
        super().__init__()

        scale = dim ** 0.5
        self.dim = dim
        self.grid_size = image_size // patch_size
        self.patch_size = patch_size
        self.n_tokens = n_tokens

        # position embedding
        self.pos_emb = nn.Parameter(torch.zeros(n_tokens, dim))

        # mask token embedding and class embedding
        self.mask_token_emb = nn.Parameter(torch.zeros(1, dim))
        self.class_emb = nn.Parameter(torch.zeros(1, dim))
        self.mask_token_pos_emb = nn.Parameter(torch.zeros(self.grid_size ** 2 + 1, dim))

        # transformer blocks
        self.ln_pre = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            ResidualAttention(dim, n_heads)
            for _ in range(n_layers)
        ])
        self.ln_post = nn.LayerNorm(dim)

        # out
        self.out = nn.Sequential(
            nn.Conv2d(dim, patch_size * patch_size * out_channels, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )

        # weights initialization
        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=scale)
        nn.init.trunc_normal_(self.mask_token_emb, std=scale)
        nn.init.trunc_normal_(self.class_emb, std=scale)
        nn.init.trunc_normal_(self.mask_token_pos_emb, std=scale)

    @property
    def z_channels(self):
        return self.dim

    def forward(self, x: Tensor):
        # reshape to 1D tokens
        B, D, H, W = x.shape
        assert H == 1 and W == self.n_tokens
        x = x.reshape(B, D, W).permute(0, 2, 1)                                                         # (B, K, D)
        x = x + self.pos_emb.unsqueeze(0)                                                               # (B, K, D)
        # mask token embedding
        mask_tokens = self.mask_token_emb.unsqueeze(0).repeat(B, self.grid_size ** 2, 1)                # (B, L, D)
        # prepend class embedding and add position embedding
        mask_tokens = torch.cat([self.class_emb.unsqueeze(0).expand(B, -1, -1), mask_tokens], dim=1)    # (B, L+1, D)
        mask_tokens = mask_tokens + self.mask_token_pos_emb.unsqueeze(0)                                # (B, L+1, D)
        # concatentate
        x = torch.cat([mask_tokens, x], dim=1)                                                          # (B, L+K+1, D)
        # transformer blocks (note nn.MultiheadAttention uses `batch_first=False` by default)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in self.blocks:
            x = block(x)
        x = x.permute(1, 0, 2)                                                                          # (B, L+K+1, D)
        # output
        x = x[:, 1:1+self.grid_size ** 2]                                                               # (B, L, D)
        x = self.ln_post(x)                                                                             # (B, L, D)
        x = x.permute(0, 2, 1).reshape(B, -1, self.grid_size, self.grid_size)                           # (B, D, H, W)
        x = self.out(x)
        return x

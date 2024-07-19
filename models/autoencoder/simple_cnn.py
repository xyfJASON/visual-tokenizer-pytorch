"""
Simple convolutional encoder and decoder with residual blocks.
"""

import torch.nn as nn
from torch import Tensor

from .base import BaseEncoder, BaseDecoder


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: Tensor):
        x = x + self.blocks(x)
        return x


class Encoder(BaseEncoder):
    def __init__(
            self,
            img_channels: int = 3,
            hidden_dim: int = 256,
            n_resblocks: int = 2,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        self.n_resblocks = n_resblocks

        self.downblocks = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.resblocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(n_resblocks)
        ])

    @property
    def z_channels(self):
        return self.hidden_dim

    @property
    def downsample_factor(self):
        return 4

    def forward(self, x: Tensor):
        x = self.downblocks(x)
        x = self.resblocks(x)
        return x


class Decoder(BaseDecoder):
    def __init__(
            self,
            img_channels: int = 3,
            hidden_dim: int = 256,
            n_resblocks: int = 2,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        self.n_resblocks = n_resblocks

        self.resblocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(n_resblocks)
        ])
        self.upblocks = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    @property
    def z_channels(self):
        return self.hidden_dim

    @property
    def upsample_factor(self):
        return 4

    def forward(self, x: Tensor):
        x = self.resblocks(x)
        x = self.upblocks(x)
        return x

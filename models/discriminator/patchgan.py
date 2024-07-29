"""
PatchGAN Discriminator

Reference:
  - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L539
"""

import torch.nn as nn
from torch import Tensor


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 64, n_layers: int = 3):
        super().__init__()

        seq = [nn.Conv2d(in_channels, dim, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]

        cur_dim = dim
        for i in range(1, n_layers):
            out_dim = dim * min(2 ** i, 8)
            seq.extend([
                nn.Conv2d(cur_dim, out_dim, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, True),
            ])
            cur_dim = out_dim

        out_dim = dim * min(2 ** n_layers, 8)
        seq.extend([
            nn.Conv2d(cur_dim, out_dim, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, True),
        ])

        seq.append(nn.Conv2d(out_dim, 1, 4, stride=1, padding=1))
        self.seq = nn.Sequential(*seq)

    def forward(self, x: Tensor):
        return self.seq(x)

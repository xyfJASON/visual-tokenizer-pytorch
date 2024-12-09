"""
StyleGAN Discriminator

References:
  - https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L639
  - https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.py#L168
  - https://github.com/google-research/maskgit/blob/main/maskgit/nets/discriminator.py#L120
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, (list, tuple)):
        up = (up, up)
    if not isinstance(down, (list, tuple)):
        down = (down, down)
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    out = upfirdn2d_native(x, kernel, *up, *down, *pad)
    return out


def upfirdn2d_native(x, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = x.shape
    x = x.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    out = x.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            Blur(blur_kernel, pad=(2, 2)),
            nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Sequential(
            Blur(blur_kernel, pad=(1, 1)),
            nn.Conv2d(in_channel, out_channel, 1, stride=2, padding=0, bias=False),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class StyleGANDiscriminator(nn.Module):
    def __init__(self, image_size, channel_multiplier=2, blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [
            nn.Conv2d(3, channels[image_size], 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channel = channels[image_size]
        for i in range(int(math.log(image_size, 2)), 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channel, channels[4], 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4] * 4 * 4, channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels[4], 1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x

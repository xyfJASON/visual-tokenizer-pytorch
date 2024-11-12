"""Finite scalar quantizer.

References:
    - https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb
    - https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/finite_scalar_quantization.py
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def round_ste(z: Tensor):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels: List[int]):
        super().__init__()

        levels = torch.tensor(levels)
        self.register_buffer('levels', levels, persistent=False)

        basis = torch.cumprod(torch.tensor([1] + levels[:-1].tolist()), dim=0, dtype=torch.long)
        self.register_buffer('basis', basis, persistent=False)

        implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))
        self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)

    @property
    def codebook(self):
        return self.implicit_codebook

    @property
    def codebook_dim(self):
        return len(self.levels)

    @property
    def codebook_num(self):
        """Number of codes in the codebook."""
        return torch.sum(self.levels).item()

    @property
    def codebook_size(self):
        """Size of the codebook."""
        return torch.prod(self.levels).item()

    def bound(self, z: Tensor, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)  # type: ignore
        shift = torch.atanh(offset / half_l)  # note: google's code incorrectly uses tan()
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: Tensor):
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        # renormalize to [-1, 1]
        half_width = self.levels // 2
        return quantized / half_width

    def forward(self, z: Tensor, return_perplexity: bool = True):
        B, C, H, W = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)

        # quantize & straight-through estimator
        quantized_z = self.quantize(flat_z).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # calculate the perplexity to monitor training
        indices, perplexity = None, None
        if return_perplexity:
            with torch.no_grad():
                flat_quantized_z = quantized_z.permute(0, 2, 3, 1).reshape(-1, C)
                indices = self.codes_to_indexes(flat_quantized_z)
                # use codebook_size instead of codebook_num to calculate the "true" perplexity
                indices_one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
                probs = torch.mean(indices_one_hot, dim=0)
                perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))

        return dict(
            quantized_z=quantized_z, indices=indices, perplexity=perplexity,
            loss_vq=torch.tensor(0.0, device=z.device, requires_grad=True),
            loss_commit=torch.tensor(0.0, device=z.device, requires_grad=True),
        )

    def _scale_and_shift(self, zhat_normalized: Tensor):
        # scale and shift to range [0, ..., L-1]
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Tensor):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis).sum(dim=-1).long()

    def indexes_to_codes(self, indices: Tensor):
        """Inverse of `codes_to_indexes`."""
        indices = indices[..., None]
        codes_non_centered = (indices // self.basis) % self.levels
        return self._scale_and_shift_inverse(codes_non_centered)

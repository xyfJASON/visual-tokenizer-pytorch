"""Lookup-free quantizer.

References:
    - https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py
    - https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/vqvae/lookup_free_quantize.py#L122
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LookupFreeQuantizer(nn.Module):
    def __init__(self, dim: int, use_entropy_reg: bool = True, entropy_reg_temp: float = 0.01):
        super().__init__()
        self.dim = dim
        self.use_entropy_reg = use_entropy_reg
        self.entropy_reg_temp = entropy_reg_temp

        basis = torch.tensor([2 ** i for i in range(dim)], dtype=torch.long)
        self.register_buffer('basis', basis, persistent=False)

        implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))
        self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)

    @property
    def codebook(self):
        return self.implicit_codebook

    @property
    def codebook_dim(self):
        return self.dim

    @property
    def codebook_num(self):
        return 2 * self.dim

    @property
    def codebook_size(self):
        return 2 ** self.dim

    @staticmethod
    def quantize(z: Tensor):
        return torch.sign(z)

    def forward(self, z: Tensor, return_perplexity: bool = True):
        # quantize
        quantized_z = self.quantize(z)

        # calculate losses
        loss_commit = F.mse_loss(z, quantized_z.detach())
        loss_entropy = None
        if self.use_entropy_reg:
            flat_z = z.permute(0, 2, 3, 1).reshape(-1, self.dim)
            dists = -2 * torch.mm(flat_z, self.codebook.T)
            loss_entropy = self.entropy_loss(dists)

        # straight-through estimator
        quantized_z = z + (quantized_z - z).detach()

        # calculate the perplexity to monitor training
        indices, perplexity = None, None
        if return_perplexity:
            with torch.no_grad():
                flat_quantized_z = quantized_z.permute(0, 2, 3, 1).reshape(-1, self.dim)
                indices = self.codes_to_indexes(flat_quantized_z)
                indices_one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
                probs = torch.mean(indices_one_hot, dim=0)
                perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))

        return dict(
            quantized_z=quantized_z, indices=indices, perplexity=perplexity,
            loss_commit=loss_commit, loss_entropy=loss_entropy,
            loss_vq=torch.tensor(0.0, device=z.device, requires_grad=True),
        )

    def codes_to_indexes(self, codes: Tensor):
        codes = codes.clamp(0, 1)
        return (codes * self.basis).sum(dim=-1).long()

    def indexes_to_codes(self, indices: Tensor):
        indices = indices[..., None]
        codes = (indices // self.basis) % 2
        return (codes * 2 - 1).float()

    def entropy_loss(self, dists: Tensor, eps: float = 1e-5):
        affinity = -dists / self.entropy_reg_temp
        probs = F.softmax(affinity, dim=1)
        log_probs = F.log_softmax(affinity + eps, dim=1)
        target_probs = probs
        avg_probs = torch.mean(target_probs, dim=0)
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))
        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=1))
        loss = sample_entropy - avg_entropy
        return loss

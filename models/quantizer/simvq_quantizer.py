"""SimVQ quantizer.

References:
    - https://github.com/youngsheen/SimVQ/blob/main/taming/modules/vqvae/quantize.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimVQQuantizer(nn.Module):
    def __init__(self, codebook_num: int, codebook_dim: int):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim

        self.codebook = nn.Embedding(codebook_num, codebook_dim)
        nn.init.normal_(self.codebook.weight, mean=0, std=self.codebook_dim ** -0.5)
        for p in self.codebook.parameters():
            p.requires_grad = False

        self.codebook_proj = nn.Linear(self.codebook_dim, self.codebook_dim)

    @property
    def codebook_size(self):
        return self.codebook_num

    def forward(self, z: Tensor, return_perplexity: bool = True):
        B, C, H, W = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)

        # project codebook
        projected_codebook_weight = self.codebook_proj(self.codebook.weight)

        # quantize
        dists = (torch.sum(flat_z ** 2, dim=1, keepdim=True) +
                 torch.sum(projected_codebook_weight ** 2, dim=1) -
                 2 * torch.mm(flat_z, projected_codebook_weight.T))
        indices = torch.argmin(dists, dim=1)
        quantized_z = F.embedding(indices, projected_codebook_weight)
        quantized_z = quantized_z.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # calculate losses
        loss_vq = F.mse_loss(z.detach(), quantized_z)
        loss_commit = F.mse_loss(z, quantized_z.detach())

        # straight-through estimator
        quantized_z = z + (quantized_z - z).detach()

        # calculate the perplexity to monitor training
        perplexity = None
        if return_perplexity:
            indices_one_hot = F.one_hot(indices, num_classes=self.codebook_num).float()
            probs = torch.mean(indices_one_hot, dim=0)
            perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))

        return dict(
            quantized_z=quantized_z, indices=indices, perplexity=perplexity,
            loss_vq=loss_vq, loss_commit=loss_commit,
        )

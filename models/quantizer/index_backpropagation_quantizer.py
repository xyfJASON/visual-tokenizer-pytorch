"""Index Backpropagation Quantizer.

References:
    - https://github.com/TencentARC/SEED-Voken/blob/main/src/IBQ/modules/vqvae/quantize.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IndexBackpropagationQuantizer(nn.Module):
    def __init__(
            self,
            codebook_num: int,
            codebook_dim: int,
            use_entropy_reg: bool = False,
            entropy_reg_temp: float = 0.01,
    ):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim
        self.use_entropy_reg = use_entropy_reg
        self.entropy_reg_temp = entropy_reg_temp

        self.codebook = nn.Embedding(codebook_num, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_num, 1 / self.codebook_num)

    @property
    def codebook_size(self):
        return self.codebook_num

    def forward(self, z: Tensor, return_perplexity: bool = True):
        B, C, H, W = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)

        # quantize
        logits = torch.mm(flat_z, self.codebook.weight.T)
        indices_soft = torch.softmax(logits, dim=1)
        _, indices = torch.max(indices_soft, dim=1)
        quantized_z_prime = self.codebook(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # calculate losses
        loss_vq = F.mse_loss(z.detach(), quantized_z_prime)
        loss_commit = F.mse_loss(z, quantized_z_prime.detach())
        loss_entropy = None
        if self.use_entropy_reg:
            loss_entropy = self.entropy_loss(logits)

        # straight-through estimator
        indices_hard = F.one_hot(indices, num_classes=self.codebook_num).float()
        indices_hard_st = indices_hard - indices_soft.detach() + indices_soft
        quantized_z = torch.mm(indices_hard_st, self.codebook.weight).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # double quantization loss
        loss_vq = loss_vq + F.mse_loss(quantized_z, z)

        # calculate the perplexity to monitor training
        perplexity = None
        if return_perplexity:
            indices_one_hot = F.one_hot(indices, num_classes=self.codebook_num).float()
            probs = torch.mean(indices_one_hot, dim=0)
            perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))

        return dict(
            quantized_z=quantized_z, indices=indices, perplexity=perplexity,
            loss_vq=loss_vq, loss_commit=loss_commit, loss_entropy=loss_entropy,
        )

    def entropy_loss(self, logits: Tensor, eps: float = 1e-5):
        affinity = logits / self.entropy_reg_temp
        probs = F.softmax(affinity, dim=1)
        log_probs = F.log_softmax(affinity + eps, dim=1)
        target_probs = probs
        avg_probs = torch.mean(target_probs, dim=0)
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))
        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=1))
        loss = sample_entropy - avg_entropy
        return loss

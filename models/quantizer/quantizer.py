import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseQuantizer


class VectorQuantizer(BaseQuantizer):
    def __init__(
            self,
            codebook_num: int,
            codebook_dim: int,
            l2_norm: bool = False,
            use_ema_update: bool = False,
            ema_decay: float = 0.99,
    ):
        super().__init__()
        self._codebook_num = codebook_num
        self._codebook_dim = codebook_dim
        self.norm_fn = lambda x: F.normalize(x, p=2, dim=1) if l2_norm else x
        self.use_ema_update = use_ema_update
        self.ema_decay = ema_decay

        self._codebook = nn.Embedding(codebook_num, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_num, 1 / self.codebook_num)

        if self.use_ema_update:
            self.ema_sumz = nn.Parameter(self.codebook.weight.clone())
            self.ema_sumn = nn.Parameter(torch.zeros((codebook_num, )))

    @property
    def codebook(self):
        return self._codebook

    @property
    def codebook_num(self):
        return self._codebook_num

    @property
    def codebook_size(self):
        return self._codebook_num

    @property
    def codebook_dim(self):
        return self._codebook_dim

    def forward(self, z: Tensor, return_perplexity: bool = True):
        B, C, H, W = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)

        # normalize
        flat_z = self.norm_fn(flat_z)
        codebook_weight = self.norm_fn(self.codebook.weight)

        # quantize
        dists = (torch.sum(flat_z ** 2, dim=1, keepdim=True) +
                 torch.sum(codebook_weight ** 2, dim=1) -
                 2 * torch.mm(flat_z, codebook_weight.T))
        indices = torch.argmin(dists, dim=1)
        quantized_z = self.norm_fn(self.codebook(indices)).reshape(B, H, W, C).permute(0, 3, 1, 2)

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

    @torch.no_grad()
    def update_codebook(self, new_sumz: Tensor, new_sumn: Tensor):
        zero_mask = torch.eq(new_sumn, 0)  # no features are assigned to these codes in this batch, should not update
        self.ema_sumz.data.copy_(self.ema_sumz.data * self.ema_decay + new_sumz * (1 - self.ema_decay))
        self.ema_sumn.data.copy_(self.ema_sumn.data * self.ema_decay + new_sumn * (1 - self.ema_decay))
        new_codes = self.ema_sumz / self.ema_sumn[:, None]
        new_codes = torch.where(zero_mask[:, None], self.codebook.weight.data, new_codes)
        self.codebook.weight.data.copy_(new_codes)


class ResidualQuantizer(nn.Module):
    def __init__(self, depth: int, codebook_num: int, codebook_dim: int, ema_decay: float = 0.99):
        super().__init__()
        self.depth = depth
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(codebook_num, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_num, 1 / self.codebook_num)

        self.ema_sumz = nn.Parameter(self.codebook.weight.clone())
        self.ema_sumn = nn.Parameter(torch.zeros((codebook_num, )))

    def quantize(self, z: Tensor):
        B, C, H, W = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)
        dists = (torch.sum(flat_z ** 2, dim=1, keepdim=True) +
                 torch.sum(self.codebook.weight ** 2, dim=1) -
                 2 * torch.mm(flat_z, self.codebook.weight.T))
        indices = torch.argmin(dists, dim=1)
        quantized_z = self.codebook(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return quantized_z, indices

    def forward(self, z: Tensor):
        quantized_z_list, indices_list = [], []
        # quantize
        for i in range(self.depth):
            quantized_z, indices = self.quantize(z)
            quantized_z_list.append(quantized_z)
            indices_list.append(indices)
            z = z - quantized_z
        # prefix sum
        for i in range(1, self.depth):
            quantized_z_list[i] = quantized_z_list[i] + quantized_z_list[i - 1]
        # straight-through estimator
        quantized_z_st = quantized_z_list[-1] + (z - quantized_z_list[-1]).detach()
        # calculate the perplexity of embeddings to monitor training
        indices = torch.cat(indices_list, dim=0)
        indices_one_hot = F.one_hot(indices, num_classes=self.codebook_num).float()
        probs = torch.mean(indices_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))
        return quantized_z_list, quantized_z_st, indices_list, perplexity

    @torch.no_grad()
    def update_codebook(self, new_sumz, new_sumn):
        zero_mask = torch.eq(new_sumn, 0)  # no features are assigned to these codes in this batch, should not update
        self.ema_sumz.data.copy_(self.ema_sumz.data * self.ema_decay + new_sumz * (1 - self.ema_decay))
        self.ema_sumn.data.copy_(self.ema_sumn.data * self.ema_decay + new_sumn * (1 - self.ema_decay))
        new_codes = self.ema_sumz / self.ema_sumn[:, None]
        new_codes = torch.where(zero_mask[:, None], self.codebook.weight.data, new_codes)
        self.codebook.weight.data.copy_(new_codes)

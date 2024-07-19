import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_num: int, codebook_dim: int):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim

        self.codebook = nn.Embedding(codebook_num, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_num, 1 / self.codebook_num)

    def forward(self, z: Tensor):
        B, C, H, W = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)

        dists = (torch.sum(flat_z ** 2, dim=1, keepdim=True) +
                 torch.sum(self.codebook.weight ** 2, dim=1) -
                 2 * torch.mm(flat_z, self.codebook.weight.T))
        indices = torch.argmin(dists, dim=1)

        quantized_z = self.codebook(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)
        quantized_z_st = z + (quantized_z - z).detach()

        # Calculate the perplexity of embeddings to monitor training
        indices_one_hot = F.one_hot(indices, num_classes=self.codebook_num).float()
        probs = torch.mean(indices_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(torch.clamp(probs, 1e-10))))

        return quantized_z, quantized_z_st, indices, perplexity


class VectorQuantizerEMA(VectorQuantizer):
    def __init__(self, codebook_num: int, codebook_dim: int, ema_decay: float = 0.99):
        super().__init__(codebook_num, codebook_dim)
        self.ema_decay = ema_decay
        self.ema_sumz = nn.Parameter(self.codebook.weight.clone())
        self.ema_sumn = nn.Parameter(torch.zeros((codebook_num, )))

    @torch.no_grad()
    def update_codebook(self, new_sumz, new_sumn):
        zero_mask = torch.eq(new_sumn, 0)  # no features are assigned to these codes in this batch, should not update
        self.ema_sumz.data.copy_(self.ema_sumz.data * self.ema_decay + new_sumz * (1 - self.ema_decay))
        self.ema_sumn.data.copy_(self.ema_sumn.data * self.ema_decay + new_sumn * (1 - self.ema_decay))
        new_codes = self.ema_sumz / self.ema_sumn[:, None]
        new_codes = torch.where(zero_mask[:, None], self.codebook.weight.data, new_codes)
        self.codebook.weight.data.copy_(new_codes)

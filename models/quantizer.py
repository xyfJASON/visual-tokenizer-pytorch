import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VectorQuantizer(nn.Module):
    def __init__(
            self,
            codebook_num: int,
            codebook_dim: int,
            l2_norm: bool = False,
            use_ema_update: bool = False,
            ema_decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim
        self.norm_fn = lambda x: F.normalize(x, p=2, dim=1) if l2_norm else x
        self.use_ema_update = use_ema_update
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(codebook_num, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_num, 1 / self.codebook_num)

        if self.use_ema_update:
            self.ema_sumz = nn.Parameter(self.codebook.weight.clone())
            self.ema_sumn = nn.Parameter(torch.zeros((codebook_num, )))

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

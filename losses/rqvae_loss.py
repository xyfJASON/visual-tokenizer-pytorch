from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .adversarial import HingeLoss
from .lpips import LPIPSLoss


class RQVAELoss(nn.Module):
    def __init__(
            self,
            disc: nn.Module,
            coef_rec: float = 1.0,
            coef_lpips: float = 1.0,
            coef_commit: float = 0.25,
            start_adv: int = 0,
            coef_adv: float = 0.75,
    ):
        super().__init__()
        self.disc = disc

        self.lpips_loss = LPIPSLoss()
        self.hinge_loss = HingeLoss(discriminator=disc)

        self.coef_rec = coef_rec
        self.coef_lpips = coef_lpips
        self.coef_commit = coef_commit
        self.start_adv = start_adv
        self.coef_adv = coef_adv

    @staticmethod
    def reconstruction_loss(recx: Tensor, x: Tensor):
        return F.mse_loss(x, recx)

    def perceptual_loss(self, recx: Tensor, x: Tensor):
        return self.lpips_loss(x, recx).mean()

    @staticmethod
    def commitment_loss(z_e: Tensor, z_q_list: List[Tensor]):
        return sum(F.mse_loss(z_e, z_q.detach()) for z_q in z_q_list)

    # RQ-VAE uses EMA to update the codebook, so the VQ loss is not needed
    # @staticmethod
    # def vq_loss(z_e: Tensor, z_q: Tensor):
    #     return F.mse_loss(z_e.detach(), z_q)

    def adversarial_loss(self, recx: Tensor):
        return self.hinge_loss('G', recx)

    def forward(self, step: int, recx: Tensor, x: Tensor, z_e: Tensor, z_q_list: List[Tensor]):
        loss_rec = self.reconstruction_loss(recx, x)
        loss_lpips = self.perceptual_loss(recx, x)
        loss_commit = self.commitment_loss(z_e, z_q_list)

        if step < self.start_adv:
            loss_adv = torch.tensor(0.0, device=recx.device, requires_grad=True)
        else:
            loss_adv = self.adversarial_loss(recx)

        loss = (
            self.coef_rec * loss_rec +
            self.coef_lpips * loss_lpips +
            self.coef_commit * loss_commit +
            self.coef_adv * loss_adv
        )

        return dict(
            loss=loss,
            loss_rec=loss_rec,
            loss_lpips=loss_lpips,
            loss_commit=loss_commit,
            loss_adv=loss_adv,
        )

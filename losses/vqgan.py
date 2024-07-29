import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .adversarial import HingeLoss
from .lpips import LPIPSLoss


class VQGANLoss(nn.Module):
    def __init__(
            self,
            disc: nn.Module,
            type_rec: str = 'l2',
            coef_rec: float = 1.0,
            coef_lpips: float = 1.0,
            coef_commit: float = 0.25,
            coef_vq: float = 1.0,
            start_adv: int = 0,
            coef_adv: float = 0.4,
            adaptive_adv_weight: bool = False,
    ):
        super().__init__()
        self.disc = disc

        self.lpips_loss = LPIPSLoss()
        self.hinge_loss = HingeLoss(discriminator=disc)

        self.type_rec = type_rec
        self.coef_rec = coef_rec
        self.coef_lpips = coef_lpips
        self.coef_commit = coef_commit
        self.coef_vq = coef_vq
        self.start_adv = start_adv
        self.coef_adv = coef_adv
        self.adaptive_adv_weight = adaptive_adv_weight

    def reconstruction_loss(self, recx: Tensor, x: Tensor):
        if self.type_rec == 'l2':
            return F.mse_loss(x, recx)
        elif self.type_rec == 'l1':
            return F.l1_loss(x, recx)
        else:
            raise ValueError(f'Unknown reconstruction loss type: {self.type_rec}')

    def perceptual_loss(self, recx: Tensor, x: Tensor):
        return self.lpips_loss(x, recx).mean()

    @staticmethod
    def commitment_loss(z_e: Tensor, z_q: Tensor):
        return F.mse_loss(z_e, z_q.detach())

    @staticmethod
    def vq_loss(z_e: Tensor, z_q: Tensor):
        return F.mse_loss(z_e.detach(), z_q)

    def adversarial_loss(self, recx: Tensor):
        return self.hinge_loss('G', recx)

    @staticmethod
    def calc_adaptive_weight(loss_nll: Tensor, loss_adv: Tensor, last_layer: Tensor):
        nll_grads = torch.autograd.grad(loss_nll, last_layer, retain_graph=True)[0]
        adv_grads = torch.autograd.grad(loss_adv, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(adv_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, step: int, recx: Tensor, x: Tensor, z_e: Tensor, z_q: Tensor, last_layer: Tensor):
        loss_rec = self.reconstruction_loss(recx, x)
        loss_lpips = self.perceptual_loss(recx, x)
        loss_commit = self.commitment_loss(z_e, z_q)
        loss_vq = self.vq_loss(z_e, z_q)

        if step < self.start_adv:
            loss_adv = torch.tensor(0.0, device=recx.device, requires_grad=True)
        else:
            loss_adv = self.adversarial_loss(recx)
            if self.adaptive_adv_weight:
                loss_nll = self.coef_rec * loss_rec + self.coef_lpips * loss_lpips
                adaptive_weight = self.calc_adaptive_weight(loss_nll, loss_adv, last_layer)
                loss_adv = adaptive_weight * loss_adv

        loss = (
            self.coef_rec * loss_rec +
            self.coef_lpips * loss_lpips +
            self.coef_commit * loss_commit +
            self.coef_vq * loss_vq +
            self.coef_adv * loss_adv
        )

        return dict(
            loss=loss,
            loss_rec=loss_rec,
            loss_lpips=loss_lpips,
            loss_commit=loss_commit,
            loss_vq=loss_vq,
            loss_adv=loss_adv,
        )

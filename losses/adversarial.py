import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdversarialLoss(nn.Module):
    """Adversarial loss for GANs

    Non-saturating loss:
        Objective of the discriminator: min -E[log(D(x))] + E[log(1-D(G(z)))]
        Objective of the generator: min -E[log(D(G(z)))]

    Hinge loss:
        Objective of the discriminator: min E[max(0, 1-D(x))] + E[max(0, 1+D(G(z)))]
        Objective of the generator: min -E[D(G(z))]

    Supports LeCam regularization on the discriminator.

    """
    def __init__(
            self,
            discriminator: nn.Module,
            loss_type: str,
            coef_lecam_reg: float = 0.0,
            lecam_reg_ema_decay: float = 0.999,
    ):
        super().__init__()
        assert loss_type in ['ns', 'hinge']

        self.discriminator = discriminator
        self.loss_type = loss_type

        self.coef_lecam_reg = coef_lecam_reg
        if self.coef_lecam_reg > 0.0:
            self.lecam_reg_ema_decay = lecam_reg_ema_decay
            self.register_buffer('ema_real_logits_mean', torch.zeros(1))
            self.register_buffer('ema_fake_logits_mean', torch.zeros(1))

    def lecam_reg(self, real_logits_mean: Tensor, fake_logits_mean: Tensor):
        lecam_loss = (torch.mean(torch.pow(F.relu(real_logits_mean - self.ema_fake_logits_mean), 2)) +
                      torch.mean(torch.pow(F.relu(self.ema_real_logits_mean - fake_logits_mean), 2)))
        self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_reg_ema_decay + real_logits_mean.detach() * (1 - self.lecam_reg_ema_decay)  # noqa
        self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_reg_ema_decay + fake_logits_mean.detach() * (1 - self.lecam_reg_ema_decay)  # noqa
        return lecam_loss

    def forward_G(self, fake_data: Tensor, *args, **kwargs):
        fake_logits = self.discriminator(fake_data, *args, **kwargs)
        if self.loss_type == 'ns':
            loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        elif self.loss_type == 'hinge':
            loss = -torch.mean(fake_logits)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')
        return loss

    def forward_D(self, fake_data: Tensor, real_data: Tensor, *args, **kwargs):
        fake_logits = self.discriminator(fake_data.detach(), *args, **kwargs)
        real_logits = self.discriminator(real_data, *args, **kwargs)

        if self.loss_type == 'ns':
            loss = (F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)) +
                    F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))) / 2
        elif self.loss_type == 'hinge':
            loss = torch.mean(F.relu(1 - real_logits) + F.relu(1 + fake_logits)) / 2
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        if self.coef_lecam_reg > 0.0:
            loss = loss + self.coef_lecam_reg * self.lecam_reg(real_logits.mean(), fake_logits.mean())

        return loss

    def forward(self, mode: str, fake_data: Tensor, real_data: Tensor = None, *args, **kwargs):
        if mode == 'G':
            return self.forward_G(fake_data, *args, **kwargs)
        elif mode == 'D':
            return self.forward_D(fake_data, real_data, *args, **kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

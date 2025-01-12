"""Perceptual loss.

References:
  - https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/perceptual_loss.py
  - https://github.com/markweberdev/maskbit/blob/main/modeling/modules/perceptual_loss.py
"""

import torch
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50'):
        super().__init__()
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
        elif model_name == "convnext_s":
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()
        else:
            raise ValueError(f'Unsupported model name: {model_name}')

        self.register_buffer('imagenet_mean', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('imagenet_std', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            image1: A tensor of shape (B, C, H, W) in range [0, 1].
            image2: A tensor of shape (B, C, H, W) in range [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        image1 = F.interpolate(image1, size=224, mode='bilinear', align_corners=False, antialias=True)
        image2 = F.interpolate(image2, size=224, mode='bilinear', align_corners=False, antialias=True)

        image1 = (image1 - self.imagenet_mean) / self.imagenet_std
        image2 = (image2 - self.imagenet_mean) / self.imagenet_std

        pred1 = self.model(image1)
        pred2 = self.model(image2)

        loss = F.mse_loss(pred1, pred2, reduction='mean')
        return loss

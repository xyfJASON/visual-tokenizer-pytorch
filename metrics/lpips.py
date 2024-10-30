import lpips
import torch
import torch.nn as nn
from torch import Tensor


def reduce_tensor(x: Tensor, reduction: str):
    assert reduction in ['mean', 'sum', 'none'], f"Reduction {reduction} not implemented"
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()
    return x


def check_range(image: Tensor, low: float, high: float):
    return torch.ge(image, low).all() and torch.le(image, high).all()


class LPIPS(nn.Module):
    def __init__(self, net: str = 'alex', reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.lpips_fn = lpips.LPIPS(net=net, verbose=False).eval()

    def __call__(self, img1: Tensor, img2: Tensor):
        """
        Args:
            img1: Tensor of shape (B, C, H, W) and dtype float32 in range [0, 1]
            img2: Tensor of shape (B, C, H, W) and dtype float32 in range [0, 1]

        Returns:
            if reduction is 'none', returns a Tensor of shape (B, )
            else, returns a scalar Tensor
        """
        assert img1.shape == img2.shape
        assert check_range(img1, 0, 1) and check_range(img2, 0, 1)

        lpips_tensor = self.lpips_fn(img1, img2, normalize=True).view(-1)
        return reduce_tensor(lpips_tensor, self.reduction)

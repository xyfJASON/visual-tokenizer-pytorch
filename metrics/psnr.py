from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
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


def image_float_to_uint8(image: torch.Tensor):
    """ [0, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert check_range(image, 0, 1)
    return (image * 255).to(dtype=torch.uint8)


class PSNR:
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction

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
        assert img1.device == img2.device
        assert check_range(img1, 0, 1) and check_range(img2, 0, 1)
        device = img1.device

        img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
        img2 = img2.permute(0, 2, 3, 1).cpu().numpy()

        psnr_list = [psnr(im1, im2) for im1, im2 in zip(img1, img2)]
        psnr_list = torch.tensor(psnr_list, device=device)
        return reduce_tensor(psnr_list, self.reduction)

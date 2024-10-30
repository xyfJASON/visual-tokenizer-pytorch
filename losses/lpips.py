import lpips
import torch.nn as nn
from torch import Tensor


class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg', verbose=False).eval()

    def forward(self, x: Tensor, y: Tensor):
        return self.lpips(x, y)

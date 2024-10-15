from abc import ABC, abstractmethod

import torch.nn as nn


class BaseQuantizer(nn.Module, ABC):

    @property
    @abstractmethod
    def codebook_dim(self):
        pass

    @property
    @abstractmethod
    def codebook_num(self):
        pass

    @property
    @abstractmethod
    def codebook_size(self):
        pass

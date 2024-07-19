from abc import ABC, abstractmethod

import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def z_channels(self):
        pass

    @property
    @abstractmethod
    def downsample_factor(self):
        pass


class BaseDecoder(nn.Module, ABC):
    @property
    @abstractmethod
    def z_channels(self):
        pass

    @property
    @abstractmethod
    def upsample_factor(self):
        pass

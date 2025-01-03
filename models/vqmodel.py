import torch.nn as nn
from torch import Tensor


class VQModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, quantizer: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.enc2code = nn.Conv2d(encoder.z_channels, quantizer.codebook_dim, 1)
        self.code2dec = nn.Conv2d(quantizer.codebook_dim, decoder.z_channels, 1)

    @property
    def codebook(self):
        return self.quantizer.codebook

    @property
    def codebook_dim(self):
        return self.quantizer.codebook_dim

    @property
    def codebook_num(self):
        return self.quantizer.codebook_num

    @property
    def codebook_size(self):
        return self.quantizer.codebook_size

    @property
    def last_layer(self):
        return self.decoder.last_layer

    def forward(self, x: Tensor):
        z = self.encoder(x)
        z = self.enc2code(z)
        out = self.quantizer(z)
        decx = self.code2dec(out['quantized_z'])
        decx = self.decoder(decx)
        return dict(decx=decx, z=z, **out)

    def decode(self, z: Tensor):
        decx = self.code2dec(z)
        decx = self.decoder(decx)
        return decx

    def decode_indices(self, indices: Tensor):
        z = self.codebook(indices)
        decx = self.code2dec(z)
        decx = self.decoder(decx)
        return decx

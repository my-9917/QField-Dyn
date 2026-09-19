"""Elementwise trainable calibration of already encoded input angles."""
import torch
from torch import nn


class AffineAngles(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(shape, dtype=torch.float64))

    def forward(self, angles):
        return self.weight * angles + self.bias

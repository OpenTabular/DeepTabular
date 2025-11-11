import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SNLinear(nn.Module):
    """Separate linear layers for each feature embedding."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("SNLinear requires a 3D input (batch, features, embedding).")
        if x.shape[-(self.weight.ndim - 1) :] != self.weight.shape[:-1]:
            raise ValueError("Input shape mismatch with weight dimensions.")

        x = x.transpose(0, 1) @ self.weight
        return x.transpose(0, 1) + self.bias

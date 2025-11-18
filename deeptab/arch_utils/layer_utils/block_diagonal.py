import torch
import torch.nn as nn


class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        if out_features % num_blocks != 0:
            raise ValueError("out_features must be divisible by num_blocks")

        block_out_features = out_features // num_blocks

        self.blocks = nn.ModuleList([nn.Linear(in_features, block_out_features, bias=bias) for _ in range(num_blocks)])

    def forward(self, x):
        x = [block(x) for block in self.blocks]
        x = torch.cat(x, dim=-1)
        return x

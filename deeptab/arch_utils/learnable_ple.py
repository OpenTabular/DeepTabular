import torch
import torch.nn as nn


class PeriodicLinearEncodingLayer(nn.Module):
    def __init__(self, bins=10, learn_bins=True):
        super().__init__()
        self.bins = bins
        self.learn_bins = learn_bins

        if self.learn_bins:
            # Learnable bin boundaries
            self.bin_boundaries = nn.Parameter(torch.linspace(0, 1, self.bins + 1))
        else:
            self.bin_boundaries = torch.linspace(-1, 1, self.bins + 1)

    def forward(self, x):
        if self.learn_bins:
            # Ensure bin boundaries are sorted
            sorted_bins = torch.sort(self.bin_boundaries)[0]
        else:
            sorted_bins = self.bin_boundaries

        # Initialize z with zeros
        z = torch.zeros(x.size(0), self.bins, device=x.device)

        for t in range(1, self.bins + 1):
            b_t_1 = sorted_bins[t - 1]
            b_t = sorted_bins[t]
            mask1 = x < b_t_1
            mask2 = x >= b_t
            mask3 = (x >= b_t_1) & (x < b_t)

            z[mask1.squeeze(), t - 1] = 0
            z[mask2.squeeze(), t - 1] = 1
            z[mask3.squeeze(), t - 1] = (x[mask3] - b_t_1) / (b_t - b_t_1)

        return z

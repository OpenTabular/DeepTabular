# ruff: noqa

import torch
import torch.nn as nn


class LearnableFourierFeatures(nn.Module):
    def __init__(self, num_features=64, d_model=512):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_features, d_model))
        self.phases = nn.Parameter(torch.randn(num_features) * 2 * torch.pi)

    def forward(self, input):
        B, K, D = input.shape
        positions = torch.arange(K, device=input.device).unsqueeze(1)
        encoding = torch.sin(positions * self.freqs.T + self.phases)
        return input + encoding.unsqueeze(0).expand(B, K, -1)


class LearnableFourierMask(nn.Module):
    def __init__(self, sequence_length, keep_ratio=0.5):
        super().__init__()
        cutoff_index = int(sequence_length * keep_ratio)
        self.mask = nn.Parameter(torch.ones(sequence_length))
        self.mask[cutoff_index:] = 0  # Start with a low-frequency cutoff

    def forward(self, input):
        B, K, D = input.shape
        freq_repr = torch.fft.fft(input, dim=1)
        masked_freq = freq_repr * self.mask.unsqueeze(1)  # Apply learnable mask
        return torch.fft.ifft(masked_freq, dim=1).real


class LearnableRandomPositionalPerturbation(nn.Module):
    def __init__(self, num_features=64, d_model=512):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_features))
        self.amplitude = nn.Parameter(torch.tensor(0.1))

    def forward(self, input):
        B, K, D = input.shape
        positions = torch.arange(K, device=input.device).unsqueeze(1)
        random_features = torch.sin(positions * self.freqs.T)
        perturbation = random_features.unsqueeze(0).expand(B, K, D) * self.amplitude
        return input + perturbation


class LearnableRandomProjection(nn.Module):
    def __init__(self, d_model=512, projection_dim=64):
        super().__init__()
        self.projection_matrix = nn.Parameter(torch.randn(d_model, projection_dim))

    def forward(self, input):
        return torch.einsum("bkd,dp->bkp", input, self.projection_matrix)


class PositionalInvariance(nn.Module):
    def __init__(self, config, invariance_type, seq_len, in_channels=None):
        super().__init__()
        # Select the appropriate layer based on config.invariance_type
        if invariance_type == "lfm":  # Learnable Fourier Mask
            self.layer = LearnableFourierMask(sequence_length=seq_len, keep_ratio=getattr(config, "keep_ratio", 0.5))
        elif invariance_type == "lff":  # Learnable Fourier Features
            self.layer = LearnableFourierFeatures(num_features=seq_len, d_model=config.d_model)
        elif invariance_type == "lprp":  # Learnable Positional Random Perturbation
            self.layer = LearnableRandomPositionalPerturbation(num_features=seq_len, d_model=config.d_model)
        elif invariance_type == "lrp":  # Learnable Random Projection
            self.layer = LearnableRandomProjection(
                d_model=config.d_model,
                projection_dim=getattr(config, "projection_dim", 64),
            )

        elif invariance_type == "conv":
            self.layer = nn.Conv1d(
                in_channels=in_channels,  # type: ignore
                out_channels=in_channels,  # type: ignore
                kernel_size=config.d_conv,
                padding=config.d_conv - 1,
                bias=config.conv_bias,
                groups=in_channels,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown positional invariance type: {config.invariance_type}")

    def forward(self, input):
        # Pass the input through the selected layer
        return self.layer(input)

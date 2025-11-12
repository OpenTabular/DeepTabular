import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, j, dim, method="linear"):
        super().__init__()
        self.j = j
        self.dim = dim
        self.method = method

        if self.method == "linear":
            # Use nn.Linear approach
            self.layer = nn.Linear(dim, j * dim)
        elif self.method == "embedding":
            # Use nn.Embedding approach
            self.layer = nn.Embedding(dim, j * dim)
        elif self.method == "conv1d":
            # Use nn.Conv1d approach
            self.layer = nn.Conv1d(in_channels=dim, out_channels=j * dim, kernel_size=1)
        else:
            raise ValueError(f"Unsupported method '{method}' for reshaping.")

    def forward(self, x):
        batch_size = x.shape[0]

        if self.method == "linear" or self.method == "embedding":
            x_reshaped = self.layer(x)  # shape: (batch_size, j * dim)
            x_reshaped = x_reshaped.view(batch_size, self.j, self.dim)  # shape: (batch_size, j, dim)
        elif self.method == "conv1d":
            # For Conv1d, add dummy dimension and reshape
            x = x.unsqueeze(-1)  # Add dummy dimension for convolution
            x_reshaped = self.layer(x)  # shape: (batch_size, j * dim, 1)
            x_reshaped = x_reshaped.squeeze(-1)  # Remove dummy dimension
            x_reshaped = x_reshaped.view(batch_size, self.j, self.dim)  # shape: (batch_size, j, dim)

        return x_reshaped  # type: ignore


class AttentionNetBlock(nn.Module):
    def __init__(
        self,
        channels,
        in_channels,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward,
        transformer_activation,
        output_dim,
        attn_dropout,
        layer_norm_eps,
        norm_first,
        bias,
        activation,
        embedding_activation,
        norm_f,
        method,
    ):
        super().__init__()

        self.reshape = Reshape(channels, in_channels, method)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            activation=transformer_activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=norm_f,
        )

        self.linear = nn.Linear(d_model, output_dim)
        self.activation = activation
        self.embedding_activation = embedding_activation

    def forward(self, x):
        z = self.reshape(x)
        x = self.embedding_activation(z)
        x = self.encoder(x)
        x = z + x
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        x = self.activation(x)
        return x

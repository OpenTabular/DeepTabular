import torch.nn as nn

from ..get_norm_fn import get_normalization_layer
from .mamba_arch import ResidualBlock


class MambAttn(nn.Module):
    """Mamba model composed of alternating MambaBlocks and Attention layers.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of alternating ResidualBlock (Mamba layers) and
        attention layers constituting the model.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # Define Mamba and Attention layers alternation
        self.layers = nn.ModuleList()

        total_blocks = config.n_layers + config.n_attention_layers  # Total blocks to be created
        attention_count = 0

        for i in range(total_blocks):
            # Insert attention layer after N Mamba layers
            if (i + 1) % (config.n_mamba_per_attention + 1) == 0:
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.attn_dropout,
                    )
                )
                attention_count += 1
            else:
                self.layers.append(
                    ResidualBlock(
                        d_model=config.d_model,
                        expand_factor=config.expand_factor,
                        bias=config.bias,
                        d_conv=config.d_conv,
                        conv_bias=config.conv_bias,
                        dropout=config.dropout,
                        dt_rank=config.dt_rank,
                        d_state=config.d_state,
                        dt_scale=config.dt_scale,
                        dt_init=config.dt_init,
                        dt_max=config.dt_max,
                        dt_min=config.dt_min,
                        dt_init_floor=config.dt_init_floor,
                        norm=get_normalization_layer(config),  # type: ignore
                        activation=config.activation,
                        bidirectional=config.bidirectional,
                        use_learnable_interaction=config.use_learnable_interaction,
                        layer_norm_eps=config.layer_norm_eps,
                        AD_weight_decay=config.AD_weight_decay,
                        BC_layer_norm=config.BC_layer_norm,
                        use_pscan=config.use_pscan,
                    )
                )

        # Check the type of the last layer and append the desired one if necessary
        if config.last_layer == "attn":
            if not isinstance(self.layers[-1], nn.MultiheadAttention):
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.dropout,
                    )
                )
        else:
            if not isinstance(self.layers[-1], ResidualBlock):
                self.layers.append(
                    ResidualBlock(
                        d_model=config.d_model,
                        expand_factor=config.expand_factor,
                        bias=config.bias,
                        d_conv=config.d_conv,
                        conv_bias=config.conv_bias,
                        dropout=config.dropout,
                        dt_rank=config.dt_rank,
                        d_state=config.d_state,
                        dt_scale=config.dt_scale,
                        dt_init=config.dt_init,
                        dt_max=config.dt_max,
                        dt_min=config.dt_min,
                        dt_init_floor=config.dt_init_floor,
                        norm=get_normalization_layer(config),  # type: ignore
                        activation=config.activation,
                        bidirectional=config.bidirectional,
                        use_learnable_interaction=config.use_learnable_interaction,
                        layer_norm_eps=config.layer_norm_eps,
                        AD_weight_decay=config.AD_weight_decay,
                        BC_layer_norm=config.BC_layer_norm,
                        use_pscan=config.use_pscan,
                    )
                )

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                # If it's an attention layer, handle input shape (seq_len, batch, embed_dim)
                # Switch to (seq_len, batch, embed_dim) for attention
                x = x.transpose(0, 1)
                x, _ = layer(x, x, x)
                # Switch back to (batch, seq_len, embed_dim)
                x = x.transpose(0, 1)
            else:
                # Otherwise, pass through Mamba block
                x = layer(x)

        return x

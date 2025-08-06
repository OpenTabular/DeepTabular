from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class DefaultTrambularConfig:
    """Configuration class for the Default Mambular model with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree parameters.
    dt_init : str, default="random"
        Initialization method for decision tree parameters.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="RMSNorm"
        Type of normalization used ('RMSNorm', etc.).
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).
    n_frequencies : int, default=48
        Number of frequencies for PLR embeddings.
    frequencies_init_scale : float, default=0.01
        Initial scale for frequency parameters in embeddings.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding layers.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before being passed to Mamba layers.
    head_layer_sizes : list, default=()
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    pooling_method : str, default="avg"
        Pooling method to use ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token to the input sequences.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2', 'mamba-triton').
    """

    # Optimizer Parameters
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Architecture Parameters
    d_model: int = 64  # see Appendix B in the Trambular paper
    n_layers: int = 6
    d_conv: int = 4
    dilation: int = 1
    expand_factor: int = 2
    bias: bool = False
    dropout: float = 0.0
    dt_rank: str = "auto"
    d_state: int = 32  # see Appendix B in the Trambular paper
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    norm: str = "RMSNorm"
    activation: Callable = nn.SiLU()  # noqa: RUF009
    layer_norm_eps: float = 1e-05
    conv_bias: bool = False
    AD_weight_decay: bool = True
    BC_layer_norm: bool = False

    # Embedding Parameters
    embedding_activation: Callable = nn.Identity()  # noqa: RUF009
    embedding_type: str = "linear"
    embedding_bias: bool = False
    plr_lite: bool = False
    n_frequencies: int = 48
    frequencies_init_scale: float = 0.01
    layer_norm_after_embedding: bool = False
    shuffle_embeddings: bool = False

    # Head Parameters
    n_heads: int = 8
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Additional Features
    pooling_method: str = "avg"
    bidirectional: bool = False
    use_learnable_interaction: bool = False
    use_cls: bool = False
    use_pscan: bool = False
    # use_accelerator: bool = False

    # Mamba Version
    mamba_version: str = "mamba-torch"

    def __post_init__(self):
        # no idea why the linter is not working here, but it is not needed for now
        # if self.mamba_version not in ["mamba-torch", "mamba1", "mamba2", "mamba-triton"]:
        #    raise ValueError(f"Invalid mamba_version: {self.mamba_version}. "
        #                     "Choose from 'mamba-torch', 'mamba-triton', 'mamba1', or 'mamba2'.")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive integers.")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate and weight decay must be non-negative.")

        if self.mamba_version == "mamba-triton" and (self.d_model % 2 != 0 or self.d_state % 2 != 0):
            raise ValueError("d_model and d_state must be even integers for proper tensor operations.")

        if self.mamba_version == "mamba-triton" and not torch.cuda.is_available():
            raise ValueError(
                "Mamba Triton requires CUDA, but it is not available. "
                "Please check your environment or use a different Mamba version."
            )

        if self.mamba_version == "mamba-triton" and self.use_pscan:
            raise ValueError(
                "Accelerator cannot be used with PSCAN. "
                "Please set use_pscan to False if you want to use the accelerator."
            )

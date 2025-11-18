# Source: https://github.com/Qwicen/node
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_aware_initialization import ModuleWithInit
from .layer_utils.sparsemax import sparsemax, sparsemoid
from .numpy_utils import check_numpy


class ODST(ModuleWithInit):
    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_dim=1,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
    ):
        """Oblivious Differentiable Sparsemax Trees (ODST).

        ODST is a differentiable module for decision tree-based models, where each tree
        is trained using sparsemax to compute feature weights and sparsemoid to compute
        binary leaf weights. This class is designed as a drop-in replacement for `nn.Linear` layers.

        Parameters
        ----------
        in_features : int
            Number of features in the input tensor.
        num_trees : int
            Number of trees in this layer.
        depth : int, optional
            Number of splits (depth) in each tree. Default is 6.
        tree_dim : int, optional
            Number of output channels for each tree's response. Default is 1.
        flatten_output : bool, optional
            If True, returns output in a flattened shape of [..., num_trees * tree_dim];
            otherwise returns [..., num_trees, tree_dim]. Default is True.
        choice_function : callable, optional
            Function that computes feature weights as a simplex, such that
            `choice_function(tensor, dim).sum(dim) == 1`. Default is `sparsemax`.
        bin_function : callable, optional
            Function that computes tree leaf weights as values in the range [0, 1].
            Default is `sparsemoid`.
        initialize_response_ : callable, optional
            In-place initializer for the response tensor in each tree. Default is `nn.init.normal_`.
        initialize_selection_logits_ : callable, optional
            In-place initializer for the feature selection logits. Default is `nn.init.uniform_`.
        threshold_init_beta : float, optional
            Initializes thresholds based on quantiles of the data using a Beta distribution.
            Controls the initial threshold distribution; values > 1 make thresholds closer to the median.
            Default is 1.0.
        threshold_init_cutoff : float, optional
            Initializer for log-temperatures, with values > 1.0 adding margin between data points
            and sparse-sigmoid cutoffs. Default is 1.0.

        Attributes
        ----------
        response : torch.nn.Parameter
            Parameter for tree responses.
        feature_selection_logits : torch.nn.Parameter
            Logits that select features for the trees.
        feature_thresholds : torch.nn.Parameter
            Threshold values for feature splits in the trees.
        log_temperatures : torch.nn.Parameter
            Log-temperatures for threshold adjustments.
        bin_codes_1hot : torch.nn.Parameter
            One-hot encoded binary codes for leaf mapping.

        Methods
        -------
        forward(input)
            Forward pass through the ODST model.
        initialize(input, eps=1e-6)
            Data-aware initialization of thresholds and log-temperatures based on input data.
        """

        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = (
            depth,
            num_trees,
            tree_dim,
            flatten_output,
        )
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2**depth]), requires_grad=True)
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, x):  # type: ignore
        """Forward pass through ODST model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape [batch_size, in_features] or higher dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, num_trees * tree_dim] if `flatten_output` is True,
            otherwise [batch_size, num_trees, tree_dim].
        """
        if len(x.shape) < 2:
            raise ValueError("Input tensor must have at least 2 dimensions")
        if len(x.shape) > 2:
            return self.forward(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum("bi,ind->bnd", x, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, x, eps=1e-6):
        """Data-aware initialization of thresholds and log-temperatures based on input data.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape [batch_size, in_features] used for threshold initialization.
        eps : float, optional
            Small value added to avoid log(0) errors in temperature initialization. Default is 1e-6.
        """
        # data-aware initializer
        if len(x.shape) != 2:
            raise ValueError("Input tensor must have 2 dimensions")
        if x.shape[0] < 1000:
            warn(  # noqa
                "Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                "You can do so manually before training. Use with torch.no_grad() for memory efficiency."
            )
        with torch.no_grad():
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum("bi,ind->bnd", x, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(
                    map(
                        np.percentile,
                        check_numpy(feature_values.flatten(1, 2).t()),
                        percentiles_q.flatten(),
                    )
                ),
                dtype=feature_values.dtype,
                device=feature_values.device,
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(
                check_numpy(abs(feature_values - self.feature_thresholds)),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.feature_selection_logits.shape[0]}, \
            num_trees={self.num_trees}, depth={self.depth}, tree_dim={self.tree_dim}, \
            flatten_output={self.flatten_output})"


class DenseBlock(nn.Sequential):
    """DenseBlock is a multi-layer module that sequentially stacks instances of `Module`,
    typically decision tree models like `ODST`. Each layer in the block produces additional features,
    enabling the model to learn complex representations.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    layer_dim : int
        Dimensionality of each layer in the block.
    num_layers : int
        Number of layers to stack in the block.
    tree_dim : int, optional
        Dimensionality of the output channels from each tree. Default is 1.
    max_features : int, optional
        Maximum dimensionality for feature expansion. If None, feature expansion is unrestricted.
        Default is None.
    input_dropout : float, optional
        Dropout rate applied to the input features of each layer during training. Default is 0.0.
    flatten_output : bool, optional
        If True, flattens the output along the tree dimension. Default is True.
    Module : nn.Module, optional
        Module class to use for each layer in the block, typically a decision tree model.
        Default is `ODST`.
    **kwargs : dict
        Additional keyword arguments for the `Module` instances.

    Attributes
    ----------
    num_layers : int
        Number of layers in the block.
    layer_dim : int
        Dimensionality of each layer.
    tree_dim : int
        Dimensionality of each tree's output in the layer.
    max_features : int or None
        Maximum feature dimensionality allowed for expansion.
    flatten_output : bool
        Determines whether to flatten the output.
    input_dropout : float
        Dropout rate applied to each layer's input.

    Methods
    -------
    forward(x)
        Performs the forward pass through the block, producing feature-expanded outputs.
    """

    def __init__(
        self,
        input_dim,
        layer_dim,
        num_layers,
        tree_dim=1,
        max_features=None,
        input_dropout=0.0,
        flatten_output=True,
        Module=ODST,
        **kwargs,
    ):
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True, **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float("inf"))
            layers.append(oddt)

        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, layer_dim, tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):  # type: ignore
        """Forward pass through the DenseBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim] or higher dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor with expanded features, where shape depends on `flatten_output`.
            If `flatten_output` is True, returns tensor of shape
            [..., num_layers * layer_dim * tree_dim].
            Otherwise, returns [..., num_layers * layer_dim, tree_dim].
        """
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs

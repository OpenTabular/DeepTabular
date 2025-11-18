import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralEmbeddingTree(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        temperature=0.0,
    ):
        """Initialize the neural decision tree with a neural network at each leaf.

        Parameters:
        -----------
        input_dim: int
            The number of input features.
        depth: int
            The depth of the tree. The number of leaves will be 2^depth.
        output_dim: int
            The number of output classes (default is 1 for regression tasks).
        lamda: float
            Regularization parameter.
        """
        super().__init__()

        self.temperature = temperature
        self.output_dim = output_dim
        self.depth = int(math.log2(output_dim))

        # Initialize internal nodes with linear layers followed by hard thresholds
        self.inner_nodes = nn.Sequential(
            nn.Linear(input_dim + 1, output_dim, bias=False),
        )

    def forward(self, X):
        """Implementation of the forward pass with hard decision boundaries."""
        batch_size = X.size()[0]
        X = self._data_augment(X)

        # Get the decision boundaries for the internal nodes
        decision_boundaries = self.inner_nodes(X)

        # Apply hard thresholding to simulate binary decisions
        if self.temperature > 0.0:
            # Replace sigmoid with Gumbel-Softmax for path_prob calculation
            logits = decision_boundaries / self.temperature
            path_prob = (logits > 0).float() + logits.sigmoid() - logits.sigmoid().detach()
        else:
            path_prob = (decision_boundaries > 0).float()

        # Prepare for routing at the internal nodes
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)

        # Iterate through internal nodes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.output_dim)

        return mu

    def _data_augment(self, X):
        return F.pad(X, (1, 0), value=1)

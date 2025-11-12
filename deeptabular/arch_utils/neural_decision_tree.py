import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        input_dim,
        depth,
        output_dim=1,
        lamda=1e-3,
        temperature=0.0,
        node_sampling=0.3,
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
        self.internal_node_num_ = 2**depth - 1
        self.leaf_node_num_ = 2**depth
        self.lamda = lamda
        self.depth = depth
        self.temperature = temperature
        self.node_sampling = node_sampling

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-d)) for d in range(0, depth)]

        # Initialize internal nodes with linear layers followed by hard thresholds
        self.inner_nodes = nn.Sequential(
            nn.Linear(input_dim + 1, self.internal_node_num_, bias=False),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_, output_dim, bias=False)

    def forward(self, X, return_penalty=False):
        if return_penalty:
            _mu, _penalty = self._penalty_forward(X)
        else:
            _mu = self._forward(X)
        y_pred = self.leaf_nodes(_mu)
        if return_penalty:
            return y_pred, _penalty  # type: ignore
        else:
            return y_pred

    def _penalty_forward(self, X):
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
        _penalty = torch.tensor(0.0)

        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _forward(self, X):
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

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """Calculate the regularization penalty by sampling a fraction of nodes with safeguards against NaNs."""
        batch_size = _mu.size(0)

        # Reshape _mu and _path_prob for broadcasting
        _mu = _mu.view(batch_size, 2**layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        # Determine sample size
        num_nodes = _path_prob.size(1)
        sample_size = max(1, int(self.node_sampling * num_nodes))

        # Randomly sample nodes for penalty calculation
        indices = torch.randperm(num_nodes)[:sample_size]
        sampled_path_prob = _path_prob[:, indices]
        sampled_mu = _mu[:, indices // 2]

        # Calculate alpha in a batched manner
        epsilon = 1e-6  # Small constant to prevent division by zero
        alpha = torch.sum(sampled_path_prob * sampled_mu, dim=0) / (torch.sum(sampled_mu, dim=0) + epsilon)

        # Clip alpha to avoid NaNs in log calculation
        alpha = alpha.clamp(epsilon, 1 - epsilon)

        # Calculate penalty with broadcasting
        coeff = self.penalty_list[layer_idx]
        penalty = -0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha)).sum()

        return penalty

    def _data_augment(self, X):
        return F.pad(X, (1, 0), value=1)

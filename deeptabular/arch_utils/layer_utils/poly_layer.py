import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


class ScaledPolynomialLayer(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

        # Initialize polynomial feature generator
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        # Initialize learnable scaling parameter
        self.weights = nn.Parameter(torch.ones(self.degree))

    def forward(self, x):
        # Scale the input to the range [-1, 1]
        x_np = x.detach().cpu().numpy()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaled = scaler.fit_transform(x_np) * 1e-05

        # Generate polynomial features
        poly_features = self.poly.fit_transform(x_scaled)

        # Convert polynomial features back to tensor
        poly_features = torch.tensor(poly_features, dtype=torch.float32).to(x.device)

        # Apply the learnable scaling parameter
        output = poly_features * self.weights

        output = torch.clamp(output, min=-1e5, max=1e3)

        return output

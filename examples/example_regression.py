# Simulate data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularRegressor

# Set random seed for reproducibility
np.random.seed(0)

# Number of samples
n_samples = 1000
n_features = 5

# Generate random features
X = np.random.randn(n_samples, n_features)
coefficients = np.random.randn(n_features)

# Generate target variable
y = np.dot(X, coefficients) + np.random.randn(n_samples)

# Create a DataFrame to store the data
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
data["target"] = y

# Split data into features and target variable
X = data.drop(columns=["target"])
y = np.array(data["target"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Instantiate the regressor
regressor = MambularRegressor()

# Fit the model on training data
regressor.fit(X_train, y_train, max_epochs=10)

print(regressor.evaluate(X_test, y_test))

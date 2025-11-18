import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deeptab.models import MambularClassifier

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
# Convert y to multiclass by categorizing into quartiles
y = pd.qcut(y, 4, labels=False)

# Create a DataFrame to store the data
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
data["target"] = y

# Split data into features and target variable
X = data.drop(columns=["target"])
y = data["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the classifier
classifier = MambularClassifier()

# Fit the model on training data
classifier.fit(X_train, y_train, max_epochs=10)

print(classifier.evaluate(X_test, y_test))

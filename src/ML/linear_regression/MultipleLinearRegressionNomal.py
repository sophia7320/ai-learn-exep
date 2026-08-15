import numpy as np


class MultipleLinearRegressionNomal:
    def __init__(self):
        self.weights = None

    def predict(self, X: np.ndarray):
        return X @ self.weights + self.bias

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

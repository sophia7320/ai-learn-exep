import numpy as np


class MultipleLinearRegressionNomal:
    def __init__(self):
        self.weights = None

    def predict(self, X: np.ndarray):
        return X @ self.weights

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

    def r_square(self, X: np.ndarray, y: np.ndarray):
        pred = self.predict(X)
        y_mean = y.mean()

        ss_res = np.sum((pred - y) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        return 1 - ss_res / ss_tot

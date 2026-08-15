import numpy as np


class LinearRegressionNomal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def predict(self, X: np.ndarray):
        return X * self.w + self.b

    def fit(self, X: np.ndarray, y: np.ndarray):
        x_mean = X.mean()
        y_mean = y.mean()

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def r_square(self, X: np.ndarray, y: np.ndarray):
        pred = self.predict(X)
        y_mean = y.mean()

        ss_res = np.sum((pred - y) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        return 1 - ss_res / ss_tot

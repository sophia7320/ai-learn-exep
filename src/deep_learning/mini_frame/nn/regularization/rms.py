import numpy as np

from ..model import Model


class RMS(Model):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.gamma = 1
        self.eps = eps

        self.gamma_d = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        self.square_sum = np.sum(X**2, axis=1, keepdims=True)
        return X / np.sqrt(self.square_sum + self.eps)

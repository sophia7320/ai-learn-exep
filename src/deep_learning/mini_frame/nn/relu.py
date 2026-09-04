import numpy as np

from .model import Model


class ReLU(Model):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, grad):
        return np.where(self.X > 0, grad, 0)

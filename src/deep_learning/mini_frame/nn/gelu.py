import numpy as np

from .model import Model


class Gelu(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return (
            0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + np.sqrt(2 / np.pi) * X)))
        )

    def backward(self, grad):
        return grad * (
            0.5
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.X + np.sqrt(2 / np.pi) * self.X)))
            + 0.5 * np.sqrt(2 / np.pi) * np.exp(-np.square(self.X))
        )

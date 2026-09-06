import numpy as np

from .model import Model


class GeLU(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        self.sigma = 1.0 / (1.0 + np.exp(-1.702 * X))
        return self.sigma * self.X

    def backward(self, grad):
        return grad * (self.sigma + self.X * self.sigma * (1 - self.sigma) * 1.702)

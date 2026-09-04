import numpy as np

from .model import Model


class Sigmoid(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return 1.0 / (1.0 + np.exp(-X))

    def backward(self, grad):
        s = self.forward(self.X)
        return grad * s * (1 - s)

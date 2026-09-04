import numpy as np

from .model import Model


class LayerNorm(Model):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        self.X = X
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.std = np.std(X, axis=-1, keepdims=True)
        return (X - self.mean) / np.sqrt(self.std + self.eps)

    def backward(self, grad):
        pass

    def __call__(self, X):
        return self.forward(X)

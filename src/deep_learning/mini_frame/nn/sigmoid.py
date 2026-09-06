import numpy as np

from .model import Model


class Sigmoid(Model):
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = np.clip(X, -500, 500)
        self.s = 1.0 / (1.0 + np.exp(-self.X))
        return self.s

    def backward(self, grad):
        # print(grad)
        grad = grad.reshape(self.X.shape)
        s = self.s
        return grad * s * (1 - s)

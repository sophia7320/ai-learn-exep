import numpy as np

from ..model import Model


class RMS(Model):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.gamma = np.ones(1)
        self.eps = eps

        self.gamma_d = np.zeros(1)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        self.squar_mean = np.mean(X**2, axis=1, keepdims=True)
        self.rms = np.sqrt(self.var + self.eps)

        self.X_hat = X / self.rms
        return self.gamma * self.X_hat

    def backward(self, grad):
        self.gamma_d[:] = np.mean(grad * self.X_hat)

        return (
            grad * self.gamma / self.rms
            - np.mean(grad * self.X_hat, axis=1, keepdims=True) * self.X_hat / self.rms
        )

    def parameters(self):
        return [(self.gamma, self.gamma_d, False)]

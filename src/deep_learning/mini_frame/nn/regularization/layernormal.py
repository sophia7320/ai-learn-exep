import numpy as np

from .model import Model


class LayerNorm(Model):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = np.ones(1)
        self.beta = np.zeros(1)
        self.gamma_d = np.zeros_like(self.gamma)
        self.beta_d = np.zeros_like(self.beta)

    def forward(self, X):
        self.X = X

        self.moving_mean = np.mean(X, axis=1, keepdims=True)
        self.moving_std = np.sqrt(np.var(X, axis=1, keepdims=True) + self.eps)

        self.X_hat = (X - self.moving_mean) / self.moving_std

        return self.gamma * self.X_hat + self.beta

    def backward(self, grad):
        # shape = (n_in, 1)
        self.gamma_d[:] = np.mean(grad * self.X_hat)
        self.beta_d[:] = np.mean(grad)

        return (
            grad * self.gamma / self.moving_std
            - np.mean(grad * self.gamma, axis=1, keepdims=True) / self.moving_std
            - np.mean(grad * self.gamma * self.X_hat, axis=1, keepdims=True)
            * self.X_hat
            / self.moving_std
        )

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        return [(self.gamma, self.gamma_d, False), (self.beta, self.beta_d, False)]

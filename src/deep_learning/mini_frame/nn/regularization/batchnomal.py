import numpy as np

from ..model import Model


class BatchNormalization(Model):
    def __init__(self, fan_in, rng=None, eps=1e-5):
        super().__init__()
        self.fan_in = fan_in
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rng = rng if not isinstance(rng, int) else np.random.default_rng(rng)

        self.eps = eps

        self.gamma = np.ones(fan_in)
        self.beta = np.zeros(fan_in)

        self.moving_mean = np.zeros(fan_in)
        self.moving_std = np.ones(fan_in)
        self.x_hat = None

        self.gamma_d = np.ones_like(self.gamma)
        self.beta_d = np.zeros_like(self.beta)

    def forward(self, X):
        self.X = X
        if self.training:
            self.moving_mean[:] = np.mean(X, axis=0, keepdims=True)
            self.moving_std[:] = np.std(X, axis=0, keepdims=True)
            self.x_hat = (X - self.moving_mean) / (self.moving_std + self.eps)
            return self.x_hat * self.gamma.reshape(1, -1) + self.beta.reshape(1, -1)

    def backward(self, grad):
        self.gamma_d[:] = np.mean(self.x_hat * grad, axis=0, keepdims=True)
        self.beta_d[:] = np.mean(grad, axis=0, keepdims=True)
        return (
            (grad - self.gamma_d * self.x_hat - self.beta_d)
            / (self.moving_std + self.eps)
            * self.gamma.reshape(1, -1)
        )

    def parameters(self):
        return [(self.gamma, self.gamma_d, False), (self.beta, self.beta_d, False)]

    def __call__(self, X):
        return self.forward(X)

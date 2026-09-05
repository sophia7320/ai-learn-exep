import numpy as np

from ..model import Model


class BatchNormalization(Model):
    def __init__(self, fan_in, rng=None, eps=1e-5):
        super().__init__()
        self.fan_in = fan_in
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rng = rng if not isinstance(rng, int) else np.random.default_rng(rng)

        self.eps = eps

        self.gamma = np.ones(size=(1, fan_in))
        self.beta = np.zeros(size=(1, fan_in))

        self.moving_mean = np.zeros(fan_in).reshape(1, -1)
        self.moving_std = np.ones(fan_in).reshape(1, -1)
        self.x_hat = None

        self.gamma_d = np.ones_like(self.gamma)
        self.beta_d = np.zeros_like(self.beta)

    def forward(self, X):
        self.X = X
        if self.training:
            # shape = (1, fan_in)
            self.moving_mean[:] = np.mean(X, axis=0, keepdims=True)
            self.moving_std[:] = np.sqrt(np.var(X, axis=0, keepdims=True) + self.eps)

            self.x_hat = (X - self.moving_mean) / self.moving_std
            return self.x_hat * self.gamma + self.beta

    def backward(self, grad):
        # n = grad.shape[0]

        # shape = (1, fan_in)
        self.gamma_d[:] = np.mean(self.x_hat * grad, axis=0, keepdims=True)
        self.beta_d[:] = np.mean(grad, axis=0, keepdims=True)

        return (
            grad * self.gamma / self.moving_std
            - np.mean(grad * self.gamma * self.x_hat, axis=0, keepdims=True)
            * self.x_hat
            / self.moving_std
            # / n :: 当使用 np.sum 时，需要除以 n (对方差计算求导得出)
            - np.mean(grad * self.gamma, axis=0, keepdims=True) / self.moving_std
            # / n :: 当使用 np.sum 时，需要除以 n (对平均数计算求导得出)
        )

    def parameters(self):
        return [(self.gamma, self.gamma_d, False), (self.beta, self.beta_d, False)]

    def __call__(self, X):
        return self.forward(X)

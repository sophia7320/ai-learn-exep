import numpy as np

from ..model import Model


class Dropout(Model):
    def __init__(self, p, rng=None):
        self.p = p

        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rng = (
            self.rng
            if not isinstance(self.rng, int)
            else np.random.default_rng(self.rng)
        )

    def __call__(self, X):
        return self.forward(X)

    def backward(self, grad):
        if self.training:
            return grad * self.mask / (1 - self.p)
        else:
            return grad

    def forward(self, X):
        if self.training:
            self.mask = self.rng.binomial(1, 1 - self.p, size=X.shape).astype(float)
            return X * self.mask
        else:
            return X

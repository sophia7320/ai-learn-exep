import numpy as np

from .optimizer import Optimizer


class AdamW(Optimizer):
    def __init__(
        self, parameters, lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01
    ):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = np.array([np.zeros_like(param) for (param, _, _) in self.parameters])
        self.v = np.array([np.zeros_like(param) for (param, _, _) in self.parameters])

    def step(self):
        super().step()
        for param, grad, is_weights in self.parameters:
            if is_weights:
                grad = grad + self.weight_decay * param

            self.m[:] = self.betas * self.m + (1 - self.betas[0]) * grad
            self.v[:] = self.betas * self.v + (1 - self.betas[1]) * grad * grad

            m_hat = self.m / (1 - self.betas[0] ** self.step)
            v_hat = self.v / (1 - self.betas[1] ** self.step)

            param[:] = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

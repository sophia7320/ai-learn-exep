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

        self.m = [np.zeros_like(param) for (param, _, _) in self.parameters]
        self.v = [np.zeros_like(param) for (param, _, _) in self.parameters]

    def step(self):
        super().step()
        for i, (param, grad, _) in enumerate(self.parameters):
            # print(f"    {param.shape}")
            self.m[i][:] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i][:] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad**2)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.steps)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.steps)

            param[:] -= self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param
            )

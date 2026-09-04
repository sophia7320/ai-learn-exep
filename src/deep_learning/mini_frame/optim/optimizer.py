import numpy as np


class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        return NotImplementedError()

    def zero_grad(self):
        for param, param_d, isWeight in self.parameters:
            param_d[:] = np.zeros_like(param_d)

    def set_lr(self, lr):
        self.lr = lr

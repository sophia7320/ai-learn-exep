import numpy as np


class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

        self.step = 0

    def step(self):
        self.step += 1

    def zero_grad(self):
        for param, param_d, isWeight in self.parameters:
            param_d.fill(0)

    def get_lr(self):
        return self.lr

    def set_lr(self, lr):
        self.lr = lr

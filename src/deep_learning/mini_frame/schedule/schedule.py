import numpy as np

from ..optim.optimizer import Optimizer


class Schedule:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.step = 0

    def step(self):
        self.step += 1
        lr = self.get_lr()
        self.optimizer.set_lr(lr)

    def get_lr(self):
        return NotImplemented

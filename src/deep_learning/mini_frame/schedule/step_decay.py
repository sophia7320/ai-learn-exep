import numpy as np

from .schedule import Schedule


class StepDecay(Schedule):
    def __init__(self, optimizer, step_size, gamma):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

        self.base_lr = self.optimizer.get_lr()

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.step // self.step_size)

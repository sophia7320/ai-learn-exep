import numpy as np

from .schedule import Schedule


class OneCircle(Schedule):
    def __init__(self, optimizer, total_steps):
        super().__init__(optimizer)
        self.total_steps = total_steps

        self.max_lr = self.optimizer.get_lr()

    def get_lr(self):
        if self.step < self.total_steps / 2:
            return self.max_lr * (self.step / (self.total_steps / 2))
        else:
            progress = (self.step - self.total_steps / 2) / (self.total_steps / 2)
            return self.max_lr * (1 - progress) + self.max_lr * 0.0001 * progress

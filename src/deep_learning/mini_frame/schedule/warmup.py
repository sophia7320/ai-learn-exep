import numpy as np

from .schedule import Schedule


class Warmup(Schedule):
    def __init__(self, optimizer, warmup_steps):
        super().__init__(optimizer)
        self.warmup_steps = 1000
        self.max_lr = self.optimizer.get_lr()

    def get_lr(self):
        return self.max_lr * min(self.step / self.warmup_steps, 1.0)

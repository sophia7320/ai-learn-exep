import numpy as np

from .schedule import Schedule


class Cosine(Schedule):
    def __init__(self, optimizer, total_steps, eta_min=0):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.eta_max = self.optimizer.get_lr()

    def get_lr(self):
        return (
            self.eta_min
            + (self.eta_max - self.eta_min)
            * (1 + np.cos(np.pi * self.step / self.total_steps))
            / 2
        )

import numpy as np


class MESLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.y_pred = None
        self.y_true = None

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.y_pred = y_pred.flatten()
        self.y_true = y_true.flatten()

        loss_sum = np.sum((self.y_pred - self.y_true) ** 2)

        if self.reduction == "mean":
            return loss_sum / len(self.y_pred)
        if self.reduction == "sum":
            return loss_sum

    def backward(self):
        grad = 2 * (self.y_pred - self.y_true) / len(self.y_pred)
        return grad

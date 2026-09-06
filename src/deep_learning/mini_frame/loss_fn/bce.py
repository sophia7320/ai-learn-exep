import numpy as np


class BCELoss:
    def __init__(self, eps=1e-10, reduction="mean"):
        self.y_hat = None
        self.y = None

        self.eps = eps
        self.reduction = reduction

    def __call__(self, y_hat, y):
        self.y_hat = y_hat.astype(np.float64).reshape(-1)
        self.y_hat = np.clip(self.y_hat, self.eps, 1 - self.eps)
        self.y = y.astype(np.float64).reshape(-1)

        self.loss_sum = np.sum(
            -self.y * np.log(self.y_hat) - (1 - self.y) * np.log(1 - self.y_hat)
        )
        if self.reduction == "sum":
            return self.loss_sum
        if self.reduction == "mean":
            return self.loss_sum / self.y.shape[0]

    def backward(self):
        # print(self.y_hat.shape, self.y.shape)

        return (
            -(self.y / self.y_hat - (1 - self.y) / (1 - self.y_hat)) / self.y.shape[0]
        )

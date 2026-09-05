from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)

    def step(self):
        super().step()
        for param, grad, _ in self.parameters:
            param -= self.lr * grad

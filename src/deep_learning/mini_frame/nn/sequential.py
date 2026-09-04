import numpy as np

from .model import Model


class Sequential(Model):
    def __init__(self, *args):
        super().__init__()
        self.models: list[Model] = [*args]

    def forward(self, X):
        for model in self.models:
            X = model.forward(X)
        return X

    def backward(self, grad):
        for model in reversed(self.models):
            grad = model.backward(grad)
        return grad

    def parameters(self):
        params = []
        for model in self.models:
            params.extend(model.parameters())
        return params

    def __call__(self, X):
        return self.forward(X)

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

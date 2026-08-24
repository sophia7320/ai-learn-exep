import numpy as np

from .layer import Layer


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, X: np.ndarray):
        current = X
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def predict(self, X: np.ndarray):
        result = self.forward(X)
        return result >= 0.5

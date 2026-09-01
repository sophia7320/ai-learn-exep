import random

from .value import Value


class Neuron:
    def __init__(self, n_inputs, weights=None, bias=None):
        self.n_inputs = n_inputs

        scale = (2.0 / self.n_inputs) ** 0.5

        if weights is None:
            self.weights = [
                Value(random.uniform(-scale, scale)) for _ in range(self.n_inputs)
            ]
        else:
            self.weights = weights

        if bias is None:
            self.bias = Value(0)
        else:
            self.bias = bias

    def __call__(self, input):
        # print(self.weights, input)
        res = sum(
            (w * x for w, x in zip(self.weights, input)), self.bias
        )  # + self.bias

        return res.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]

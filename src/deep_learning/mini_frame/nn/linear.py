import numpy as np

from .model import Model


class Linear(Model):
    def __init__(self, fan_in, fan_out, rng=None):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out

        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rng = rng if not isinstance(rng, int) else np.random.default_rng(rng)

        self.weights = self.rng.normal(
            0, np.sqrt(2 / self.fan_in), (self.fan_in, self.fan_out)
        )

        self.biases = np.zeros(self.fan_out)

        self.weights_d = np.zeros_like(self.weights)
        self.biases_d = np.zeros_like(self.biases)

    def forward(self, X):
        self.X = X
        return self.X @ self.weights + self.biases.reshape(1, -1)

    def backward(self, grad):
        n = len(grad)
        self.weights_d = self.X.T @ grad / n
        self.biases_d = np.mean(grad, axis=0)

    def parameters(self):
        return [
            (self.weights, self.weights_d, True),
            (self.biases, self.biases_d, False),
        ]

    def __call__(self, X):
        return self.forward(X)


if __name__ == "__main__":
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(np.mean(m, axis=0))

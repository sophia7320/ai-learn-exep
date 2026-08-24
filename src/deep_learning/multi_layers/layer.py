import numpy as np


class Layer:
    def sigmoid(self, y: np.ndarray) -> np.ndarray:
        y_cliped = np.clip(y, -500, 500)
        return 1 / (1 + np.exp(-y_cliped))

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        weights: np.ndarray = None,
        biases: np.ndarray = None,
        random_state: int = 42,
    ):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rng = np.random.default_rng(random_state)

        if weights is None:
            self.weights = self.rng.uniform(-0.5, 0.5, (n_neurons, n_inputs))
        elif weights.shape == (n_neurons, n_inputs):
            self.weights = weights
        else:
            raise ValueError("the weight is not matches")

        if biases is None:
            self.biases = np.zeros(n_neurons)
        elif biases.shape[0] == n_neurons:
            self.biases = biases
        else:
            raise ValueError("the bias is bad")

    def forward(self, X: np.ndarray):
        if X.shape[-1] != self.n_inputs:
            raise ValueError(
                f"input value error ,input shape is {X.shape} , n_inputs = {self.n_inputs}"
            )

        result = self.weights @ X.T
        if result.ndim == 1:
            return self.sigmoid(result + self.biases)
        else:
            return self.sigmoid(result + self.biases[:, np.newaxis]).T

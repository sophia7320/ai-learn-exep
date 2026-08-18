import numpy as np


def sigmoid(z: np.ndarray):
    z = np.clip(z, -500, 500)

    z = np.exp(-z)
    return 1 / (1 + z)


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.001):
        self.weights = None
        self.learning_rate = learning_rate

    def predict(self, X: np.ndarray, threhold: float = 0.5):
        z = X @ self.weights
        p = sigmoid(z)

        return np.where(p >= threhold, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, print_every=200):
        self.weights = np.zeros(X.shape[1])
        m = X.shape[0]

        for epoch in range(epochs):
            z = X @ self.weights
            p = sigmoid(z)
            dw = X.T @ (p - y) / m

            self.weights -= dw * self.learning_rate

            if epoch % print_every == 0:
                print(f"epoch {epoch} : current w = {self.weights}")

        return self

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        pred = self.predict(X)

        return np.sum((pred == y).astype(int)) / X.shape[0]

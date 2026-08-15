import numpy as np


class MultipleLinearRegression:
    def __init__(self, n_feature: int, learning_rate: float = 0.001):
        self.n_feature = n_feature
        self.weights = np.zeros(n_feature)
        self.learning_rate = learning_rate

    def predict(self, X: np.ndarray):
        return X @ self.weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoches: int = 50000,
        every_epoches: int = 10000,
    ):
        for epoch in range(epoches):
            pred = self.predict(X)

            dw = np.mean(X * (pred - y)[:, np.newaxis], axis=0) * 2 * self.learning_rate

            self.weights -= dw

            if epoch % every_epoches == 0:
                print(f"epoch {epoch}: cur_w = {self.weights}")


if __name__ == "__main__":
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    x = x ** np.arange(1, 4)
    print(x)

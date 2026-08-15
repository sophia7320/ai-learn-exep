import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.05):
        self.w = 0.0
        self.b = 0.0
        self.learning_rate = learning_rate

    def predict(self, X: np.ndarray):
        return X * self.w + self.b

    def compute_gradient(self, X: np.ndarray, y: np.ndarray):
        pred = self.predict(X)
        n = y.shape[0]

        dw = (2 / n) * (X * (pred - y)).sum()
        # dw = (2 / SAMPLE_N) * sum(
        #     x_act * (x_pred - y_act) for x_act, x_pred, y_act in zip(X, pred, y)
        # )
        db = (2 / n) * (pred - y).sum()
        # db = (2 / SAMPLE_N) * sum(x_pred - y_act for x_pred, y_act in zip(pred, y))

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=200, every_epoch=20):
        for epoch in range(epochs):
            dw, db = self.compute_gradient(X, y)
            self.w -= dw * self.learning_rate
            self.b -= db * self.learning_rate

            if epoch % every_epoch == 0:
                print(f"epoch {epoch} : cur_w = {self.w:.3f} , cur_b ={self.b:.3f}")

import numpy as np
import random

TRUE_W = 3.0
TRUE_B = 7.0
SAMPLE_N = 200

random.seed(42)

X = [random.uniform(0, 10) for _ in range(SAMPLE_N)]
y = [x * TRUE_W + TRUE_B + random.gauss(0, 5) for x in X]


class LinearRegression:
    def __init__(self, learning_rate=0.05):
        self.w = 0
        self.b = 0
        self.learning_rate = learning_rate

    def predict(self, X):
        return [(self.w * x + self.b) for x in X]

    def compute_gradient(self, X, y):
        pred = self.predict(X)
        dw = (2 / SAMPLE_N) * sum(
            x_act * (x_pred - y_act) for x_act, x_pred, y_act in zip(X, pred, y)
        )
        db = (2 / SAMPLE_N) * sum(x_pred - y_act for x_pred, y_act in zip(pred, y))

        return dw, db

    def fit(self, X, y, epochs=200, every_epoch=20):
        for epoch in range(epochs):
            dw, db = self.compute_gradient(X, y)
            self.w -= dw * self.learning_rate
            self.b -= db * self.learning_rate

            if epoch % every_epoch == 0:
                print(f"epoch {epoch} : cur_w = {self.w:.3} , cur_b ={self.b:.3}")


linear = LinearRegression(learning_rate=0.005)

linear.fit(X, y, epochs=10000, every_epoch=200)

print(f"True_w = {TRUE_W} True_b = {TRUE_B}")
print(f"model_w = {linear.w:.3} model_b = {linear.b:.3}")

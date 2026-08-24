import numpy as np


class TwoLayerNetwork:
    def __init__(self, learning_rate=0.001, random_state=42):
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_state)

        # failed
        self.w_hidden = np.hstack(
            [self.rng.uniform(-1, 1, 3)[:, np.newaxis] for _ in range(2)]
        )
        self.w_output = self.rng.uniform(-1, 1, 3)

        # self.w_hidden = np.hstack(
        #     [self.rng.uniform(-0.5, 0.5, 3)[:, np.newaxis] for _ in range(2)]
        # )
        # self.w_output = self.rng.uniform(-0.5, 0.5, 3)

        self.hidden_out = None

    def sigmoid(self, y: np.ndarray):
        y_cliped = np.clip(y, -500, 500)
        return 1.0 / (1.0 + np.exp(-y_cliped))

    def predict(self, X: np.ndarray):
        n_sample = X.shape[0]
        X = np.hstack([np.ones(n_sample)[:, np.newaxis], X])

        X_out_h_raw = X @ self.w_hidden
        self.hidden_out = self.sigmoid(X_out_h_raw)

        X_out_raw = (
            np.hstack([np.ones(n_sample)[:, np.newaxis], self.hidden_out])
            @ self.w_output
        )
        return self.sigmoid(X_out_raw)

    def fit(
        self, X: np.ndarray, y: np.ndarray, epochs=10001, every_print=2000, show=False
    ):
        n_sample = X.shape[0]
        X_raw = X
        X = np.hstack([np.ones(n_sample)[:, np.newaxis], X])

        for epoch in range(epochs):
            pred = self.predict(X_raw)
            # print(pred)

            d = pred - y

            hidden_out = np.hstack([np.ones(n_sample)[:, np.newaxis], self.hidden_out])

            d_output = hidden_out.T @ d / n_sample

            hy = self.hidden_out

            d_hidden = (hy * (1 - hy)) * (
                d[:, np.newaxis] * self.w_output[np.newaxis, 1:]
            )

            d_hidden = X.T @ d_hidden / n_sample

            self.w_output -= d_output * self.learning_rate
            self.w_hidden -= d_hidden * self.learning_rate

            if show and epoch % every_print == 0:
                print(
                    f"epoch {epoch} \nw_hidden = \n{self.w_hidden} w_out = {self.w_output}"
                )
                print(f"cur_pred : {pred}")
                loss = -(1 / n_sample) * np.sum(
                    y * np.log2(pred) + (1 - y) * np.log2(1 - pred)
                )
                print(f"loss = {loss}")

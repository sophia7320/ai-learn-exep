import numpy as np


class Network:
    def __init__(
        self,
        n_inputs,
        n_hidden_neurons,
        loss_fn,
        loss_deriv,
        activation_fn,
        activation_deriv,
        lr=0.1,
        random_state=42,
    ):
        rng = np.random.default_rng(random_state)

        hidden_scale = (2.0 / n_inputs) ** 0.5
        self.hidden_neurons_weights = np.hstack(
            [
                rng.uniform(-hidden_scale, hidden_scale, n_inputs)[:, np.newaxis]
                for _ in range(n_hidden_neurons)
            ]
        )
        self.hidden_neurons_bias = np.zeros(n_hidden_neurons)

        out_scale = (2.0 / n_hidden_neurons) ** 0.5

        self.out_neuron_weights = rng.uniform(-out_scale, out_scale, n_hidden_neurons)
        self.out_neuron_bias = 0

        self.loss_fn = loss_fn
        self.loss_d = loss_deriv

        self.activation_fn = activation_fn
        self.activation_d = activation_deriv

        self.lr = lr

        self.hidden_output_cache = None
        self.out_cache = None
        self.n_samples = 0

    def sigmoid(self, y):
        y = np.clip(y, -500, 500)
        return 1.0 / (1.0 + np.exp(-y))

    def forward(self, X):
        self.X = X

        self.hidden_pre = (
            X @ self.hidden_neurons_weights + self.hidden_neurons_bias[np.newaxis, :]
        )

        hidden_out = self.activation_fn(self.hidden_pre)
        self.hidden_output_cache = hidden_out

        out = hidden_out @ self.out_neuron_weights + self.out_neuron_bias

        self.out_cache = self.sigmoid(out)

        return self.out_cache

    def backward(self, target):

        d_out_z = (
            self.loss_d(self.out_cache, target) * self.out_cache * (1 - self.out_cache)
        )

        d_out_w = self.hidden_output_cache.T @ d_out_z / self.n_samples
        d_out_b = d_out_z.mean()

        # print(d_out_z, self.hidden_neurons_weights)

        d_hidden_out_z = np.outer(d_out_z, self.out_neuron_weights) * self.activation_d(
            self.hidden_pre
        )

        d_hidden_w = self.X.T @ d_hidden_out_z / self.n_samples
        d_hidden_b = np.mean(d_hidden_out_z, axis=0)

        self.out_neuron_weights -= d_out_w * self.lr
        self.out_neuron_bias -= d_out_b * self.lr
        self.hidden_neurons_weights -= d_hidden_w * self.lr
        self.hidden_neurons_bias -= d_hidden_b * self.lr

    def fit(self, X, y, epochs, print_every=100):
        self.n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs + 1):
            # print(self.hidden_neurons_weights.shape)
            pred = self.forward(X)

            # print(y.shape)
            loss = self.loss_fn(pred, y)
            losses.append(loss)

            self.backward(y)

            predict = pred >= 0.5
            accuracy = np.mean(predict == y)

            if epoch % print_every == 0:
                print(f"epoch {epoch} : loss = {loss} accuracy = {accuracy}")

        return losses

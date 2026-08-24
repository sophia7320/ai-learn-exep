from .layer import Layer
from .value import Value


def mse_loss(predict, labal):
    diff = predict + Value(-labal)

    return diff * diff


class Network:
    def __init__(self, sizes, learning_rate=0.1, random_state=42):
        self.layers = []
        self.learning_rate = learning_rate

        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1], random_state=random_state))

    def __call__(self, x):
        current = x
        for layer in self.layers:
            current = layer(current)
            current = [current] if not isinstance(current, list) else current

        return current[0] if isinstance(current, list) else current

    def parameters(self):
        p = []
        for layer in self.layers:
            p.extend(layer.parameters())

        return p

    def grad_zero(self):
        for para in self.parameters():
            para.grad = 0.0

    def fit(self, train_data: list, epochs=2000):
        n_sample = len(train_data)
        for epoch in range(epochs):
            total_loss = Value(0.0)

            for input, labal in train_data:
                pred = self(input)
                loss = mse_loss(pred, labal)
                total_loss = total_loss + loss

            total_loss = total_loss * Value(1 / n_sample)

            self.grad_zero()
            total_loss.backward()

            for p in self.parameters():
                p.data -= p.grad * self.learning_rate

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f}")

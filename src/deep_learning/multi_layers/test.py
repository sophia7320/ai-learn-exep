import numpy as np

from .layer import Layer
from .network import Network

rng = np.random.default_rng(42)


def xor_test():
    hidden = Layer(
        n_inputs=2,
        n_neurons=2,
        weights=np.array([[20.0, 20.0], [-20.0, -20.0]]),
        biases=np.array([-10.0, 30.0]),
    )

    output = Layer(
        n_inputs=2,
        n_neurons=1,
        weights=np.array([[20.0, 20.0]]),
        biases=np.array([-30.0]),
    )

    xor_net = Network([hidden, output])

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    for inputs, expected in zip(X, y):
        result = xor_net.forward(inputs)
        predicted = 1 if result[0] >= 0.5 else 0
        print(result)
        print(
            f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})"
        )


def circle_test():
    X = rng.uniform(-1, 1, (200, 2))

    x, y = X[:, 0], X[:, 1]

    label = (x * x + y * y) < 0.25

    circle_net = Network(
        [
            Layer(n_inputs=2, n_neurons=8),
            Layer(n_inputs=8, n_neurons=1),
        ]
    )

    correct = 0
    # for inputs, expected in zip(X, label):
    #     result = circle_net.forward(inputs)
    #     predicted = 1 if result[0] >= 0.5 else 0
    #     if predicted == expected:
    #         correct += 1
    result = circle_net.predict(X)
    print(result.shape, label.shape)
    correct = (result.squeeze() == label).sum()

    print(
        f"Accuracy with random weights: {correct}/{len(label)} ({100 * correct / len(label):.1f}%)"
    )


def test():
    xor_test()
    circle_test()


if __name__ == "__main__":
    test()

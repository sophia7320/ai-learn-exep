import numpy as np
from sklearn.linear_model import Perceptron as SkPerceptron

from .two_layer_network import TwoLayerNetwork

np.set_printoptions(precision=2, suppress=False)


def test():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    print("==========sklearn xor (single neuron) ===========")
    clf = SkPerceptron(max_iter=100, tol=1e-3)
    clf.fit(X, y)
    print(clf.predict(X))

    print("===========my xor=================")
    net = TwoLayerNetwork(learning_rate=2.0)
    net.fit(X, y, epochs=10000, show=True)

    pred = net.predict(X)

    result = (pred >= 0.5).astype(int)

    print(f"raw :   {pred}")
    print(f"result: {result}")

    print()


if __name__ == "__main__":
    test()

import numpy as np
from scipy.special import erf

from .network import Network


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_act(x):
    return np.tanh(x)


def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t * t


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)


def leaky_relu(x, alpha=0.01):
    return np.maximum(x, x * alpha)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x):
    phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    return phi + x * pdf


def swish(x):
    return x * sigmoid(x)


def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softmax(xs):
    max_x = np.max(xs, axis=-1, keepdims=True)
    exps = np.exp(xs - max_x)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(
        f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)"
    )


def mse(predictions, targets):
    n = len(targets)
    total = np.sum((predictions - targets) ** 2)
    return total / n


def mse_gradient(predictions, targets):
    return 2 * (predictions - targets)


def binary_cross_entropy(predictions, targets, eps=1e-15):
    p_clipped = np.clip(predictions, eps, 1 - eps)

    total = -(targets * np.log2(p_clipped) + (1 - targets) * np.log2(1 - p_clipped))
    return total.mean()


def bce_gradient(predictions, targets, eps=1e-15):
    p_clipped = np.clip(predictions, eps, 1 - eps)
    grads = -(targets / p_clipped) + (1 - targets) / (1 - p_clipped)
    return grads


def test():
    gradient_scan("Sigmoid", sigmoid_derivative)
    gradient_scan("Tanh", tanh_derivative)
    gradient_scan("ReLU", relu_derivative)
    gradient_scan("Leaky ReLU", leaky_relu_derivative)
    gradient_scan("GELU", gelu_derivative)
    gradient_scan("Swish", swish_derivative)


def make_circle_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 2))

    y = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1] < 1.5

    return X, y


def circle_test():
    X, y = make_circle_data(500)

    shelff = np.random.default_rng(42).permutation(len(y))

    X = X[shelff]
    y = y[shelff]

    split = int(len(y) * 0.8)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    configs = [
        ("Sigmoid", sigmoid, sigmoid_derivative),
        ("ReLU", relu, relu_derivative),
        ("GELU", gelu, gelu_derivative),
    ]

    print("==================MSE=======================")
    results = {}
    for name, act_fn, act_d_fn in configs:
        print(f"\n=== Training with {name} ===")
        net = Network(
            2,
            8,
            loss_fn=mse,
            loss_deriv=mse_gradient,
            activation_fn=act_fn,
            activation_deriv=act_d_fn,
            lr=0.1,
        )
        losses = net.fit(X_train, y_train, epochs=10000, print_every=2500)

        pred = net.forward(X_test) >= 0.5

        accucary = np.mean(pred == y_test)
        results[name] = (losses, accucary)

    print("\n=== Final Loss Comparison ===")
    for name, (losses, acc) in results.items():
        print(
            f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1] / losses[0]) * 100:.1f}%) accuracy: {acc:.4f}"
        )

    print("\n")

    print("================BCE===========================")
    results = {}
    for name, act_fn, act_d_fn in configs:
        print(f"\n=== Training with {name} ===")
        net = Network(
            2,
            8,
            loss_fn=binary_cross_entropy,
            loss_deriv=bce_gradient,
            activation_fn=act_fn,
            activation_deriv=act_d_fn,
            lr=0.1,
        )
        losses = net.fit(X_train, y_train, epochs=10000, print_every=1000)
        pred = net.forward(X_test) >= 0.5

        accucary = np.mean(pred == y_test)
        results[name] = (losses, accucary)

    print("\n=== Final Loss Comparison ===")
    for name, (losses, acc) in results.items():
        print(
            f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1] / losses[0]) * 100:.1f}%) accuracy: {acc:.4f}"
        )


if __name__ == "__main__":
    circle_test()

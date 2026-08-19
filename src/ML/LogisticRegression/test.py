import numpy as np

from ...util.Timer import Timer
from .ClassificationMetrics import ClassificationMetrics
from .LogisticRegression import LogisticRegression

rng = np.random.default_rng(42)

N = 2000


def generate_data(N):
    X = np.vstack([rng.normal(2, 1, (N // 2, 2)), rng.normal(5, 1, (N // 2, 2))])
    y = np.hstack([np.zeros(N // 2), np.ones(N // 2)])

    shuffled_idx = rng.permutation(N)

    X = X[shuffled_idx]
    y = y[shuffled_idx]

    print(f"Generated {N} samples (2 classes, 2 features)")
    print("Class 0 center: (2, 2), Class 1 center: (5, 5)")
    print("First 5 samples:")
    for i in range(5):
        print(f"  Features: {X[i]}, Label: {y[i]}")

    return X, y


def test():
    X, y = generate_data(N)

    X = np.hstack([np.ones(N)[:, np.newaxis], X])

    split = int(N * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(learning_rate=1)

    timer = Timer()
    timer.start()
    print("\n=== Training Logistic Regression ===")
    model.fit(X_train, y_train, epochs=100001, print_every=10000)
    print(f"time cost : {timer.stop():.4f}s")

    print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
    print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
    print(f"Weights: [{model.weights}]")

    y_pred_test = model.predict(X_test)
    print("\n=== Classification Report (Test Set) ===")
    metrics = ClassificationMetrics(y_test, y_pred_test)
    metrics.print_confusion_matrix()
    metrics.print_report()


def main():
    test()


if __name__ == "__main__":
    main()

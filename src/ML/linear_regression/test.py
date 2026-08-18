import numpy as np
from .LinearRegression import LinearRegression
from .LinearRegressionNomal import LinearRegressionNomal
from .MultipleLinearRegression import MultipleLinearRegression
from .MultipleLinearRegressionNomal import MultipleLinearRegressionNomal

np.set_printoptions(precision=2)
rng = np.random.default_rng(42)


def main():
    if True:
        test_single_linear()
    if True:
        test_mutiple_linear()


def test_single_linear():
    TRUE_W = 3.0
    TRUE_B = 7.0
    SAMPLE_N = 40000

    # random.seed(42)

    X = rng.uniform(5, 30, SAMPLE_N)

    y = X * (TRUE_W) + TRUE_B + rng.normal(0, 5, SAMPLE_N)

    linear = LinearRegression(learning_rate=0.00175)
    nomal = LinearRegressionNomal()

    linear.fit(X, y, epochs=10000, every_epoch=1000)

    print(f"True_w = {TRUE_W} True_b = {TRUE_B}")
    print(f"gradient descent model_w = {linear.w:.3f} model_b = {linear.b:.3f}")
    nomal.fit(X, y)
    print(f"nomal equation   model_w = {nomal.w:.3f}  model_b = {nomal.b:.3f}")

    print(f"r_squared = {nomal.r_square(X, y)}")


# mutiple linear regression
def test_mutiple_linear():
    SAMPLE_N = 50000
    SAMPLE_D = 6

    TRUE_W = np.array([1, 1, 4, 5, 1, 4])

    X = rng.uniform(0, 10, (SAMPLE_N, SAMPLE_D))
    y = X @ TRUE_W + rng.normal(0, 5, SAMPLE_N)

    multiple_nomal = MultipleLinearRegressionNomal()
    multiple_nomal.fit(X, y)

    print(f"True_w = {TRUE_W}")
    print(f"nomal equation   model_w = {multiple_nomal.weights.T}")

    # gradient descent
    multiple_gradient = MultipleLinearRegression(n_feature=6, learning_rate=0.002)

    multiple_gradient.fit(X, y, every_epoches=5000, epoches=10001)
    print(f"True_w = {TRUE_W}")
    print(f"multiple gradient   model_w = {multiple_gradient.weights}")
    print(f"r_squared = {multiple_gradient.r_square(X, y)}")


if __name__ == "__main__":
    main()

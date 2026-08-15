import numpy as np
from LinearRegression import LinearRegression
from LinearRegressionNomal import LinearRegressionNomal

TRUE_W = 3.0
TRUE_B = 7.0
SAMPLE_N = 80000

# random.seed(42)

rng = np.random.default_rng()

X = rng.uniform(0, 10, SAMPLE_N)

y = X * (TRUE_W) + TRUE_B + rng.normal(0, 10, SAMPLE_N)


linear = LinearRegression(learning_rate=0.005)
nomal = LinearRegressionNomal()

linear.fit(X, y, epochs=5000, every_epoch=1000)

print(f"True_w = {TRUE_W} True_b = {TRUE_B}")
print(f"gradient descent model_w = {linear.w:.3f} model_b = {linear.b:.3f}")
nomal.fit(X, y)
print(f"nomal equation   model_w = {nomal.w:.3f}  model_b = {nomal.b:.3f}")

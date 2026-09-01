import matplotlib.pyplot as plt
import numpy as np


def rmsprop(grad, lr=0.1, beta=0.9):
    res = [0] * len(grad)
    res[0] = (1 - beta) * grad[0]
    for i in range(1, len(grad)):
        res[i] = beta * res[i - 1] + (1 - beta) * grad[i]

    return np.array(res)


if __name__ == "__main__":
    x = np.linspace(-50, 50, 10000)

    grad = np.cos(x) + 1

    rms = rmsprop(grad, beta=0.5)

    # plt.subplot(1, 2, 1)  # 1行2列的第1张图
    plt.plot(x, grad / rms, label="RMSprop", color="#1f77b4", linewidth=2.5)
    plt.axhline(0, color="black", linewidth=0.5, linestyle=":")  # 画出 x 轴
    plt.axvline(0, color="black", linewidth=0.5, linestyle=":")  # 画出 y 轴
    plt.title("Activation Functions Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.show()
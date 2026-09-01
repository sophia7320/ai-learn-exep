"""RMSProp 优化器：原理 + 可视化演示

RMSProp 更新规则（对每个参数分别进行，按元素运算）:
    v_t      = beta * v_{t-1} + (1 - beta) * g_t ** 2   # EMA(g^2)，语义上是 lr 的分母
    lr_eff_t = lr / (sqrt(v_t) + eps)                   # 每个参数独立的有效学习率
    w_{t+1}  = w_t - lr_eff_t * g_t

    注意：v 既不是梯度，也不是对 grad 的修正；它只出现在 lr 的分母上，
    等价写法 w -= (lr / sqrt(v+eps)) * grad。算法上两种写法结果相同，
    但语义应读作“缩放学习率”，而不是“缩放梯度”。

为什么需要它:
    - SGD 对所有参数共用一个学习率。当损失曲面“病态”(不同方向曲率差很多)时，
      一个 lr 无法同时照顾陡峭方向和平坦方向 -> 震荡或爬不动。
    - AdaGrad 用所有历史梯度平方的累加 G_t 做缩放，G_t 只增不减，
      有效学习率 lr / sqrt(G_t) 最终趋近 0 -> 训练过早停摆。
    - RMSProp 把“累加”换成“指数移动平均”，只记住最近一段时间的梯度大小，
      所以有效学习率不会无限衰减；陡峭方向自动减小步长，平坦方向自动放大步长。

注意:
    - 状态 v 必须跨迭代保存（这正是本文件中 class RMSProp 的写法）。
      对同一时刻的梯度向量做滑动平均、或每次调用都重新算，都是错误的。
    - PyTorch 里对应 torch.optim.RMSProp(params, lr, alpha=0.99, eps=1e-8)，
      其 alpha 就是这里的 beta；Adam = RMSProp + Momentum + 偏差修正。
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class RMSProp:
    """真正的 RMSProp 优化器：每个参数维护自己的 v = EMA(g^2)，作为 lr 的分母。"""

    def __init__(self, lr: float = 0.1, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None  # EMA(g^2)：在更新式里是 lr 的分母（不是梯度）

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(grad)

        # v 只负责构造 lr 的分母：对梯度平方做指数移动平均
        self.v = self.beta * self.v + (1 - self.beta) * grad**2

        # 语义：先算出每个参数的有效学习率，再乘梯度
        denominator = np.sqrt(self.v) + self.eps
        effective_lr = self.lr / denominator
        return w - effective_lr * grad


# ---------------------------------------------------------------
# 演示 1：病态损失曲面上的优化轨迹
# f(w0, w1) = 0.05*w0^2 + 5*w1^2，两个方向的曲率相差 100 倍
# ---------------------------------------------------------------
A, B = 0.05, 5.0


def demo_loss(w: np.ndarray) -> float:
    return A * w[0] ** 2 + B * w[1] ** 2


def demo_grad(w: np.ndarray) -> np.ndarray:
    return np.array([2 * A * w[0], 2 * B * w[1]], dtype=float)


def _run(kind: str, lr: float, steps: int = 250, beta: float = 0.9):
    """用 SGD / AdaGrad / RMSProp 各跑一遍，返回轨迹和 loss 序列。"""
    w = np.array([10.0, 1.0])  # w0 在平坦方向离最优点很远，w1 在陡峭方向
    path, losses = [w.copy()], [demo_loss(w)]

    state = None
    for _ in range(steps):
        g = demo_grad(w)
        if kind == "sgd":
            dw = lr * g
        elif kind == "adagrad":
            state = g**2 if state is None else state + g**2
            effective_lr = lr / (np.sqrt(state) + 1e-8)  # 分母是累计梯度平方
            dw = effective_lr * g
        else:  # rmsprop
            state = g**2 if state is None else beta * state + (1 - beta) * g**2
            effective_lr = lr / (np.sqrt(state) + 1e-8)  # 分母是 EMA(g^2)
            dw = effective_lr * g

        w = w - dw
        path.append(w.copy())
        losses.append(demo_loss(w))

    return np.array(path), np.array(losses)


def _parse_curve_toggles(argv: list[str]) -> dict[str, bool]:
    """把命令行参数转成每条曲线的显示开关。

    支持两种写法:
        --only rmsprop           只画 rmsprop
        --only sgd,rmsprop       只画 sgd 和 rmsprop
        --no-sgd --no-adagrad    隐藏 sgd 和 adagrad（其余正常显示）
    """
    toggles = {"sgd": True, "adagrad": True, "rmsprop": True}

    if "--only" in argv:
        idx = argv.index("--only")
        names = argv[idx + 1].split(",") if idx + 1 < len(argv) else []
        toggles = {key: key in names for key in toggles}

    for key in toggles:
        if f"--no-{key}" in argv:
            toggles[key] = False

    return toggles


def demo(show: bool = True, curves: dict[str, bool] | None = None):
    """画出 RMSProp 对比图。

    show=True  : 保存到 log/rmsprop_demo.png 并弹窗显示
    show=False : 只保存图片，不弹窗（服务器 / 批量跑实验时用）
    curves     : 每条曲线的显示开关，默认三条都画，例如
                 demo(curves={"sgd": False, "adagrad": False, "rmsprop": True})
                 表示只画 RMSProp 一条曲线。
    """
    if curves is None:
        curves = {"sgd": True, "adagrad": True, "rmsprop": True}

    configs = [
        ("SGD (lr=0.05)", "sgd", 0.05),
        ("AdaGrad (lr=0.50)", "adagrad", 0.50),
        ("RMSProp (lr=0.10, beta=0.9)", "rmsprop", 0.10),
    ]
    colors = {"sgd": "tab:red", "adagrad": "tab:blue", "rmsprop": "tab:green"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_sgd, ax_ada, ax_rms, ax_lr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- 上面三张：每个算法单独一张轨迹图，互不重叠 ----
    xx, yy = np.meshgrid(np.linspace(-12, 12, 400), np.linspace(-2.5, 2.5, 400))
    zz = A * xx**2 + B * yy**2
    levels = np.logspace(-2, 2.2, 14)

    for ax, (title, kind, lr) in zip([ax_sgd, ax_ada, ax_rms], configs):
        cs = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", alpha=0.55)
        ax.clabel(cs, fmt="%.0f", fontsize=6)

        if curves.get(kind, True):
            path, _ = _run(kind, lr)
            ax.plot(path[:, 0], path[:, 1], "-", color=colors[kind], lw=2.0)
            ax.plot(path[0, 0], path[0, 1], "o", color=colors[kind], ms=8, label="start")
            ax.plot(0, 0, "+", color="black", ms=13, mew=2, label="optimum (0, 0)")
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "hidden by toggle",
                transform=ax.transAxes, ha="center", va="center", color="gray",
            )

        ax.set_xlim(-12, 12)
        ax.set_xlabel("w0 (flat direction)")
        ax.set_ylabel("w1 (steep direction)")
        ax.set_title(title)

    # ---- 第四张：有效学习率对比，双 y 轴避免两条线重叠 ----
    lr_adagrad, lr_rms = 0.5, 0.1
    beta = 0.9
    t = np.arange(1, 401)
    g = 1.0

    adagrad_lr_eff = lr_adagrad / np.sqrt(t)  # G_t = t * g^2，分母一直增大，lr 一直衰减
    v = np.zeros_like(t, dtype=float)
    rms_lr_eff = np.empty_like(t)
    for i in range(len(t)):
        v[i] = g**2 if i == 0 else beta * v[i - 1] + (1 - beta) * g**2
        rms_lr_eff[i] = lr_rms / np.sqrt(v[i] + 1e-8)  # v 是 EMA(g^2)，是 lr 的分母

    ax_lr_right = ax_lr.twinx()  # 左轴给 AdaGrad，右轴给 RMSProp

    if curves.get("adagrad", True):
        ax_lr.plot(t, adagrad_lr_eff, color="tab:blue", lw=1.8, label="AdaGrad: lr / sqrt(G_t)")
    if curves.get("rmsprop", True):
        ax_lr_right.plot(t, rms_lr_eff, color="tab:green", lw=1.8, label="RMSProp: lr / sqrt(v_t)")
        ax_lr_right.axhline(lr_rms, color="tab:green", ls="--", lw=1, alpha=0.6, label=f"lr = {lr_rms}")

    ax_lr.set_xlabel("iteration  (constant gradient g = 1)")
    ax_lr.set_ylabel("AdaGrad effective lr", color="tab:blue")
    ax_lr_right.set_ylabel("RMSProp effective lr", color="tab:green")
    ax_lr.set_title("Effective lr: AdaGrad dies, RMSProp stays alive")
    ax_lr.set_ylim(0, 0.55)
    ax_lr_right.set_ylim(0, 0.11)

    # 两个轴上的图例合并到左轴
    lines = ax_lr.get_lines() + ax_lr_right.get_lines()
    ax_lr.legend(lines, [line.get_label() for line in lines], fontsize=8)
    ax_lr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("log/rmsprop_demo.png", dpi=130)
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    import sys

    demo(
        show="--no-show" not in sys.argv,
        curves=_parse_curve_toggles(sys.argv),
    )

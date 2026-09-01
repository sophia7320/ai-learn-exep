import time
import numpy as np
import matplotlib.pyplot as plt  # 👈 纯 Python 脚本的标准导入


class NativeAnimator:
    """在独立的 GUI 窗口中动态绘制数据（支持普通 .py 脚本）"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(5, 4),
    ):  # 独立窗口可以适当调大尺寸

        if legend is None:
            legend = []

        # 开启 matplotlib 的交互模式（这是非 Jupyter 环境下画动图的关键！）
        plt.ion()

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]

        # 保存坐标轴配置参数，方便后续重复调用
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xlim, self.ylim = xlim, ylim
        self.xscale, self.yscale = xscale, yscale
        self.legend = legend

        self.X, self.Y, self.fmts = None, None, fmts

    def _config_axes(self):
        """内部方法：每次清空画布后重新配置坐标轴"""
        ax = self.axes[0]
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.legend:
            ax.legend(self.legend)
        ax.grid(True)

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 1. 擦除旧线条
        self.axes[0].cla()

        # 2. 重新绘制所有历史数据线
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)

        # 3. 重新刷上坐标轴标签和图例
        self._config_axes()

        # 4. 🔴 核心替代：不再使用 display，而是强制渲染画布并短暂停顿
        self.fig.canvas.draw()  # 强制通知底层 GUI 引擎重新渲染画布
        self.fig.canvas.flush_events()  # 处理窗口管理器事件（防止窗口假死）
        plt.pause(0.01)  # 必须停顿极短的时间，图表窗口才会真正刷新出来


# ---- 🚀 测试运行（直接在本地执行这个 .py 脚本） ----
if __name__ == "__main__":
    animator = NativeAnimator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, 20],
        ylim=[0, 1],
        legend=["train", "val"],
    )

    for epoch in range(1, 21):
        train_loss = 0.9 * (0.85**epoch)
        val_loss = train_loss + 0.1 * np.random.rand()

        animator.add(epoch, [train_loss, val_loss])
        time.sleep(0.2)  # 模拟训练耗时

    # 训练结束后，关闭交互模式，并保持窗口不销毁，等待用户手动关闭
    plt.ioff()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # 用于计算 Phi 函数

# 1. 定义核心数学函数
def phi(x):
    """标准正态分布的累积分布函数 (Phi)"""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def pdf(x):
    """标准正态分布的概率密度函数 (Phi 的导数)"""
    return np.exp(-x**2 / 2.0) / np.sqrt(2.0 * np.pi)

def sigmoid(x):
    """Sigmoid 函数"""
    return 1.0 / (1.0 + np.exp(-x))

# 2. 定义激活函数及其导数（梯度）
def gelu(x): 
    return x * phi(x)

def gelu_derivative(x): 
    return phi(x) + x * pdf(x)

def swish(x): 
    return x * sigmoid(x)

def swish_derivative(x): 
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

# 3. 准备自变量数据（x 轴范围）
x = np.linspace(-7, 7, 5000)

# 4. 创建画布，设置宽高
plt.figure(figsize=(14, 6))

# --- 左图：激活函数原函数对比 ---
plt.subplot(1, 2, 1)  # 1行2列的第1张图
plt.plot(x, gelu(x), label='GELU ($x \cdot \Phi(x)$)', color='#1f77b4', linewidth=2.5)
plt.plot(x, swish(x), label='Swish ($x \cdot \sigma(x)$)', color='#ff7f0e', linestyle='--', linewidth=2.5)
plt.axhline(0, color='black', linewidth=0.5, linestyle=':') # 画出 x 轴
plt.axvline(0, color='black', linewidth=0.5, linestyle=':') # 画出 y 轴
plt.title('Activation Functions Comparison', fontsize=12, fontweight='bold')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# --- 右图：梯度（导数）对比 ---
plt.subplot(1, 2, 2)  # 1行2列的第2张图
plt.plot(x, gelu_derivative(x), label='GELU Gradient', color='#1f77b4', linewidth=2.5)
plt.plot(x, swish_derivative(x), label='Swish Gradient', color='#ff7f0e', linestyle='--', linewidth=2.5)
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.title('Gradients (Derivatives) Comparison', fontsize=12, fontweight='bold')
plt.xlabel('x')
plt.ylabel("f '(x)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# 5. 调整布局并展示
plt.tight_layout()
plt.show()



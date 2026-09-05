# AI Engineering from Scratch — 学习笔记

课程：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)

从 ML 基础开始，动手实现核心模型，逐步深入深度学习与 AI 工程。所有模型**从零实现（仅依赖 numpy）**，并与 scikit-learn / PyTorch 对照验证。

## 进度

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1: ML Fundamentals | ✅ 完成 | 监督学习基础 |
| → ML Intro | ✅ | Nearest Centroid 分类器 + baseline + 数据划分 |
| → Linear Regression | ✅ | 梯度下降 + 解析式（单变量 / 多变量）+ R² |
| → Logistic Regression | ✅ | sigmoid + 梯度下降 + 混淆矩阵 / 精确率 / 召回率 / F1 |
| → Decision Tree | ✅ | Gini / 熵 + 剪枝约束 + 特征重要性 |
| → Random Forest | ✅ | bootstrap + `sqrt` 特征子采样 + 多数投票 |
| Phase 2: Deep Learning Core | 🔧 进行中 | |
| → Two-Layer Network | ✅ | 手写反向传播，解决 XOR（对照线性模型不可分） |
| → Multi-Layer Network | 🔧 | 矩阵化多层前向 + 手搓 XOR 权重演示（bias 位置待修） |
| → Autograd Engine | ✅ | `Value` + 拓扑排序反向传播（micrograd 风格），XOR 收敛 |
| → Activation & Loss | ✅ | 参数化网络（激活/损失可注入）+ 6 种激活函数对比实验 |
| → RMSProp Optimizer | ✅ | EMA(g²) 自适应步长 + 病态曲面轨迹 / 有效步长对比 |
| → ... | 📅 | 后续推进中 |

## 目录结构

```
ai-learn-exep/
├── src/
│   ├── util/
│   │   └── Timer.py                    # 计时工具
│   ├── ML/
│   │   ├── intro/
│   │   │   └── ml_intro.py             # Nearest Centroid + baseline + 数据划分
│   │   ├── linear_regression/
│   │   │   ├── LinearRegression.py             # 单变量 · 梯度下降
│   │   │   ├── LinearRegressionNomal.py        # 单变量 · 解析式
│   │   │   ├── MultipleLinearRegression.py     # 多变量 · 梯度下降
│   │   │   ├── MultipleLinearRegressionNomal.py# 多变量 · 解析式
│   │   │   └── test.py                 # 四种实现对比 + R²
│   │   ├── LogisticRegression/
│   │   │   ├── LogisticRegression.py   # sigmoid + 梯度下降
│   │   │   ├── ClassificationMetrics.py# 混淆矩阵 / 精确率 / 召回率 / F1
│   │   │   └── test.py                 # 二分类 + train/test + 计时
│   │   └── decision_tree/
│   │       ├── decisionTree.py         # 决策树（Gini / 熵 + 剪枝 + 特征重要性）
│   │       ├── randomforest.py         # 随机森林（bootstrap + sqrt 抽样）
│   │       └── test.py                 # iris 上与 sklearn 对照
│   └── deep_learning/
│       ├── xor_two_layers/             # 两层网络 · 手写反向传播
│       │   ├── two_layer_network.py    # 2-3-1 结构 + CE 梯度
│       │   └── test.py                 # XOR（对比线性模型不可分）
│       ├── multi_layers/               # 矩阵化多层网络
│       │   ├── layer.py                # 权重矩阵 + sigmoid 层
│       │   ├── network.py              # 层组合 + predict
│       │   └── test.py                 # 手搓 XOR 权重 + 圆环数据
│       ├── backpropagation/            # 自建 autograd 引擎（micrograd 风格）
│       │   ├── value.py                # Value：自动求导 + 拓扑排序反传
│       │   ├── neuron.py               # 神经元（Value 组合）
│       │   ├── layer.py                # 层（神经元组合）
│       │   ├── network.py              # 网络 + MSE 训练循环
│       │   └── test.py                 # XOR / 圆环 + PyTorch 对照
│       ├── activation_and_loss/        # 激活函数 & 损失函数对比实验
│       │   ├── network.py              # 参数化单隐层网络（激活/损失注入）
│       │   ├── gelu.py                 # GELU / Swish 及其导数可视化
│       │   └── test.py                 # 梯度死区扫描 + 圆数据 MSE/BCE 对照
│       ├── softmax/                    # softmax 回归（FashionMNIST + PyTorch）
│       │   ├── model.py                # 线性层 + CrossEntropyLoss 训练
│       │   └── test.py                 # FashionMNIST 分类
│       └── optimizer/                  # 优化器
│           └── rmsprop.py              # RMSProp 实现 + 病态曲面轨迹图
├── main.py
├── pyproject.toml                      # uv 项目配置（numpy / sklearn / torch）
└── README.md
```

## 测试命令

```bash
cd ~/Desktop/learning/ai-learn-exep

# ML Intro（Nearest Centroid 演示）
uv run python -m src.ML.intro.ml_intro

# 线性回归（梯度下降 vs 解析式，单变量 + 多变量）
uv run python -m src.ML.linear_regression.test

# 逻辑回归（二分类，含计时与分类指标）
uv run python -m src.ML.LogisticRegression.test

# 决策树 + 随机森林（与 sklearn 对照）
uv run python -m src.ML.decision_tree.test

# 两层网络 · XOR（对比线性模型）
uv run python -m src.deep_learning.xor_two_layers.test

# 多层网络（手搓 XOR 权重 + 圆环数据）
uv run python -m src.deep_learning.multi_layers.test

# autograd 引擎（XOR / 圆环 + PyTorch 对照）
uv run python -m src.deep_learning.backpropagation.test

#softmax img
uv run python -m src.deep_learning.softmax.test

# 激活函数 & 损失函数（梯度死区扫描 + 圆数据对照实验）
uv run python -m src.deep_learning.activation_and_loss.test

# RMSProp 优化器（病态曲面轨迹 + 有效步长对比，图保存到 log/rmsprop_demo.png）
uv run python -m src.deep_learning.optimizer.rmsprop

# mini_frame
uv run python -m src.deep_learning.mini_frame.test
```

> 使用 `python -m` 模块方式运行（相对导入要求包上下文，不能直接 `python xxx.py`）。

## 实现约定

- **截距**：单变量模型自带 `b`；多变量模型由调用方显式添加全 1 列（如 `np.hstack([np.ones((n, 1)), X])`），列位置与 `predict` 保持一致
- **神经元偏置**：bias 必须在激活函数**之前**：`σ(Wx + b)`，不是 `σ(Wx) + b`
- **激活函数导数**：反向传播时 `activation_deriv` 必须喂**激活前**的值（pre-activation），forward 中需缓存原始输入；输出层用 `sigmoid + BCE` 配套（BCE 梯度抵消 sigmoid 饱和项），避免 `sigmoid + MSE` 的梯度消失
- **数据泄漏**：train/test 拆分后，训练循环只能喂训练集——`fit(X_train, y_train)`，测试集碰过之后 test accuracy 无效
- **验证方式**：每个模型都与 scikit-learn / PyTorch 对应实现对照（准确率 / 特征重要性等）
- **随机种子**：测试数据固定 `default_rng(42)`；模型内部通过 `random_state` 参数控制，且种子只应在构造时设置一次（不要在循环内重置）

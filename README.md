# AI Engineering from Scratch — 学习笔记

课程：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)

从 ML 基础开始，动手实现核心模型，逐步深入深度学习与 AI 工程。所有模型**从零实现（仅依赖 numpy）**，并与 scikit-learn 对照验证。

## 进度

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1: ML Fundamentals | 🔧 进行中 | 监督学习基础 |
| → ML Intro | ✅ | Nearest Centroid 分类器 + baseline + 数据划分 |
| → Linear Regression | ✅ | 梯度下降 + 解析式（单变量 / 多变量）+ R² |
| → Logistic Regression | ✅ | sigmoid + 梯度下降 + 混淆矩阵 / 精确率 / 召回率 / F1 |
| → Decision Tree | ✅ | Gini / 熵 + 剪枝约束 + 特征重要性 |
| → Random Forest | ✅ | bootstrap + `sqrt` 特征子采样 + 多数投票 |
| Phase 2: Deep Learning Core | 🔧 进行中 | |
| → Two-Layer Network | ✅ | 手写反向传播，解决 XOR（对照线性模型不可分） |
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
│       ├── two_layer_network_.py       # 两层网络 · 手写反向传播
│       └── test.py                     # XOR（对比线性模型不可分）
├── main.py
├── pyproject.toml                      # uv 项目配置
└── README.md
```

## 测试命令

```bash
cd ~/Desktop/ai-learning/ai-learn-exep

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

# 多层网络
uv run python -m src.deep_learning.multi_layers.test
```

> 使用 `python -m` 模块方式运行（相对导入要求包上下文，不能直接 `python xxx.py`）。

## 实现约定

- **截距**：单变量模型自带 `b`；多变量模型由调用方显式添加全 1 列（如 `np.hstack([np.ones((n, 1)), X])`），列位置与 `predict` 保持一致
- **验证方式**：每个模型都与 scikit-learn 对应实现对照（准确率 / 特征重要性等）
- **随机种子**：测试数据固定 `default_rng(42)`；模型内部通过 `random_state` 参数控制

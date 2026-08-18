# AI Engineering from Scratch — 学习笔记

课程：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)

从 ML 基础开始，动手实现核心模型，逐步深入深度学习与 AI 工程。

## 进度

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1: ML Fundamentals | 🔧 进行中 | 监督学习基础 |
| → ML Intro | ✅ | Nearest Centroid 分类器 |
| → Linear Regression | ✅ | 梯度下降 + 解析式（单变量 / 多变量） |
| → Logistic Regression | ✅ | sigmoid + 梯度下降 + 阈值分类 |
| → ... | 📅 | 后续推进中 |

## 目录结构

```
ai-learn-exep/
├── src/
│   ├── util/
│   │   └── Timer.py                    # 计时工具
│   └── ML/
│       ├── intro/
│       │   └── ml_intro.py             # Nearest Centroid 分类器
│       ├── linear_regression/
│       │   ├── LinearRegression.py             # 单变量 · 梯度下降
│       │   ├── LinearRegressionNomal.py        # 单变量 · 解析式
│       │   ├── MultipleLinearRegression.py     # 多变量 · 梯度下降
│       │   ├── MultipleLinearRegressionNomal.py# 多变量 · 解析式
│       │   └── test.py                 # 四种实现对比 + R² 评估
│       └── LogisticRegression/
│           ├── LogisticRegression.py   # sigmoid + 梯度下降
│           └── test.py                 # 二分类 + train/test 准确率
├── main.py
├── pyproject.toml
└── README.md
```

## 测试命令

```bash
cd ~/Desktop/learning/ai-learn-exep

# ML Intro（Nearest Centroid 演示）
uv run python -m src.ML.intro.ml_intro

# 线性回归（梯度下降 vs 解析式，单变量 + 多变量）
uv run python -m src.ML.linear_regression.test

# 逻辑回归（二分类，含计时）
uv run python -m src.ML.LogisticRegression.test
```

> 使用 `python -m` 模块方式运行（相对导入要求包上下文，不能直接 `python xxx.py`）。

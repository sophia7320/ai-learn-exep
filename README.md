# AI Engineering from Scratch — 学习笔记

课程：[ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)

从 ML 基础开始，动手实现核心模型，逐步深入深度学习与 AI 工程。

## 进度

| Phase | 状态 | 内容 |
|-------|------|------|
| Phase 1: ML Fundamentals | 🔧 进行中 | 监督学习基础 |
| → ML Intro | ✅ | Nearest Centroid 分类器 |
| → Linear Regression | ✅ | 梯度下降从零实现 |
| → ... | 📅 | 后续推进中 |

## 目录结构

```
ai-learn-exep/
├── src/
│   └── ML/
│       ├── intro/
│       │   └── ml_intro.py           # Nearest Centroid 分类器
│       └── linear_regression/
│           └── LinearRegression.py   # 线性回归（梯度下降）
├── main.py
├── pyproject.toml
└── README.md
```

## 运行

```bash
uv run python src/ML/intro/ml_intro.py
uv run python src/ML/linear_regression/LinearRegression.py
```

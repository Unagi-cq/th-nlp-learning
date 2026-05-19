# TH-NLP Learning - 自然语言处理算法学习路线

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

这是一个系统性的统计学习与自然语言处理(NLP)算法学习项目，逐步实现各种机器学习算法。本项目旨在帮助学习者深入理解算法原理，并通过代码实践掌握算法实现细节。

## 📚 项目结构

```
th-nlp-learning/
├── 1-统计学习方法/
│   ├── book/                 # 相关书籍资料
│   ├── note/                 # 学习笔记（PDF）
│   └── codes/                # 算法代码实现
│       ├── 第02章 感知机/
│       ├── 第03章 k近邻法/
│       ├── 第04章 朴素贝叶斯法/
│       ├── 第05章 决策树/
│       ├── 第06章 逻辑斯蒂回归与最大熵模型/
│       ├── 第07章 支持向量机/
│       ├── 第08章 提升方法/
│       └── 第10章 隐马尔可夫模型/
├── 2-集成学习/
│   ├── note/                 # 学习笔记
│   ├── figures/              # 超参数调优可视化图表
│   ├── adaboost_model.py     # AdaBoost
│   ├── randomforest_model.py # 随机森林
│   ├── gbdt_model.py         # GBDT（网格搜索+可视化）
│   ├── xgboost_model.py      # XGBoost（网格搜索+可视化）
│   └── lightgbm_model.py     # LightGBM
├── 3-深度学习/
│   └── book/                 # 相关书籍资料
├── common/
│   ├── dataset_load.py       # 统一数据集加载器
│   └── datasets/             # 数据集文件
└── README.md
```

## 关注我们

### 三黄工作室

欢迎关注我们的微信公众号「三黄工作室」，获取更多技术分享、学习资源和算法解析！

在这里你可以找到：
- 📘 深度技术文章解析
- 📚 系统化的学习路线
- 🔥 最新AI技术动态
- 💡 实用编程技巧

<div align="center">

![三黄工作室二维码](images/qr_code.png)

</div>

## 📖 算法实现进度

### 统计学习方法（第1-10章）

| 章节 | 算法 | 状态 | 说明 |
|------|------|------|------|
| 第02章 | 感知机（Perceptron） | ✅ | 原始形式实现，带可视化 |
| 第03章 | k近邻法（KNN） | ✅ | KD-Tree实现，支持低维高效搜索 |
| 第04章 | 朴素贝叶斯法 | ✅ | 多项式模型，拉普拉斯平滑 |
| 第05章 | 决策树 | ✅ | ID3（信息增益）、C4.5（信息增益率）、CART（基尼指数） |
| 第06章 | 逻辑斯蒂回归与最大熵模型 | ✅ | 二分类、多分类模型（手写+sklearn对比） |
| 第07章 | 支持向量机 | ✅ | 线性SVM、非线性SVM（RBF/多项式/Sigmoid核） |
| 第08章 | 提升方法 | ✅ | AdaBoost、GBDT（手写实现+sklearn对比） |
| 第10章 | 隐马尔可夫模型 | ✅ | 前向后向算法、维特比解码、中文分词应用 |

### 集成学习（Ensemble Learning）

| 算法 | 状态 | 说明 |
|------|------|------|
| AdaBoost | ✅ | 集成多个弱分类器，逐步调整样本权重 |
| 随机森林（Random Forest） | ✅ | Bagging + 随机特征选择 |
| GBDT | ✅ | 梯度提升决策树，支持网格搜索超参数调优与可视化 |
| XGBoost | ✅ | 极端梯度提升，支持网格搜索超参数调优与可视化 |
| LightGBM | ✅ | 基于直方图的轻量级梯度提升 |

### 深度学习

| 内容 | 状态 |
|------|------|
| - | 🚧 努力更新中... |

## 🔧 运行方式

### 安装依赖

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 安装 uv（如果未安装）
pip install uv

# 同步依赖
uv sync
```

## 💻 环境要求

- Python >= 3.11
- matplotlib >= 3.10
- numpy >= 2.2
- scikit-learn >= 1.7

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

⭐ 如果这个项目对你有帮助，请给我们一个Star！

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题或建议，请通过以下方式联系我们：
- 提交 Issue
- 关注微信公众号「三黄工作室」
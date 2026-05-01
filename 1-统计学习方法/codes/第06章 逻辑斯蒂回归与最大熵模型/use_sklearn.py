"""
use_sklearn.py

使用 sklearn 库中的 LogisticRegression 进行二分类与多分类测试

Date: 2026-03-14
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def test_binary_sklearn():
    """
    测试 sklearn 的二分类逻辑回归
    """
    print("\n" + "=" * 80)
    print("测试 sklearn.linear_model.LogisticRegression (二分类)")
    print("=" * 80)
    
    # 生成二分类数据集
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    print(f"类别分布 - 训练集: {np.bincount(y_train)}, 测试集: {np.bincount(y_test)}")
    
    # 创建并训练模型
    # penalty='l2' 是默认的正则化方式
    # C=1.0 是正则化强度的倒数（C 越大，正则化越弱）
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 预测
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    print(f"\n模型参数:")
    print(f"  权重 w: {clf.coef_[0]}")
    print(f"  偏置 b: {clf.intercept_[0]}")
    
    print("\n分类报告 (测试集):")
    print(classification_report(y_test, y_test_pred, target_names=['Class 0', 'Class 1']))
    
    # 可视化决策边界
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', alpha=0.7)
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    plt.legend()
    plt.title(f"sklearn Logistic Regression (Binary)\nTest Acc: {test_acc:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


def test_multinomial_sklearn():
    """
    测试 sklearn 的多分类逻辑回归
    """
    print("\n" + "=" * 80)
    print("测试 sklearn.linear_model.LogisticRegression (多分类)")
    print("=" * 80)
    
    # 生成多分类数据集（三分类）
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    print(f"类别分布 - 训练集: {np.bincount(y_train)}, 测试集: {np.bincount(y_test)}")
    
    # multi_class='multinomial' 用于真正的多项逻辑回归（需要 solver='lbfgs' 或 'newton-cg'）
    clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    print(f"\n模型参数:")
    print(f"  权重矩阵 W 形状: {clf.coef_.shape}")
    print(f"  偏置向量 b 形状: {clf.intercept_.shape}")
    
    for i, coef in enumerate(clf.coef_):
        print(f"  类别 {i} 的权重: {coef}")
    print(f"  类别偏置: {clf.intercept_}")
    
    print("\n分类报告 (测试集):")
    print(classification_report(y_test, y_test_pred))
    
    # 可视化决策边界
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green']
    labels = ['Class 0', 'Class 1', 'Class 2']
    
    for class_idx in range(3):
        mask = y == class_idx
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[class_idx],
            label=labels[class_idx],
            alpha=0.7
        )
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    
    plt.legend()
    plt.title(f"sklearn Logistic Regression (Multinomial)\nTest Acc: {test_acc:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_binary_sklearn()
    test_multinomial_sklearn()

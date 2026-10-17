"""
use_models.py

测试手写的逻辑斯蒂回归模型：BinaryLogisticRegression 和 MultinomialLogisticRegression

Date: 2026-03-14
"""

import sys
import os

current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)

import numpy as np
from collections import Counter
from logistic_regression import BinaryLogisticRegression, MultinomialLogisticRegression


def create_binary_dataset(n_samples=100):
    """
    创建二分类数据集
    
    参数:
        n_samples: 每个类别的样本数量
    
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    np.random.seed(42)
    
    # 类 0：中心在 (2, 2)
    X0 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    y0 = np.zeros(n_samples, dtype=int)
    
    # 类 1：中心在 (4, 4)
    X1 = np.random.randn(n_samples, 2) * 0.5 + np.array([4, 4])
    y1 = np.ones(n_samples, dtype=int)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    # 打乱数据
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    return X, y


def create_multinomial_dataset(n_samples=50):
    """
    创建多分类数据集（三分类）
    
    参数:
        n_samples: 每个类别的样本数量
    
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    np.random.seed(42)
    
    centers = np.array([
        [2, 2],
        [5, 2],
        [3.5, 4.5],
    ])
    
    X_list = []
    y_list = []
    
    for class_idx, center in enumerate(centers):
        X_c = np.random.randn(n_samples, 2) * 0.4 + center
        y_c = np.full(n_samples, class_idx, dtype=int)
        X_list.append(X_c)
        y_list.append(y_c)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    return X, y


def test_binary_logistic(X, y):
    """
    测试二项逻辑回归
    """
    print("\n" + "=" * 80)
    print("测试 BinaryLogisticRegression (二分类)")
    print("=" * 80)
    
    model = BinaryLogisticRegression(learning_rate=0.3, max_iter=1000)
    model.fit(X, y)
    
    preds = model.predict(X)
    acc = np.mean(preds == y)
    
    print(f"\n训练集准确率: {acc:.4f}")
    print(f"权重: {model.w}")
    print(f"偏置: {model.b}")
    
    # 测试预测
    test_X = np.array([
        [2.5, 2.5],  # 应该预测为 0
        [4.5, 4.5],  # 应该预测为 1
    ])
    test_preds = model.predict(test_X)
    print(f"\n测试预测:")
    for x, pred in zip(test_X, test_preds):
        print(f"  {x} -> Class {pred} (概率: {model.predict_proba(x):.4f})")
    
    return model


def test_multinomial_logistic(X, y):
    """
    测试多项逻辑回归
    """
    print("\n" + "=" * 80)
    print("测试 MultinomialLogisticRegression (多分类)")
    print("=" * 80)
    
    model = MultinomialLogisticRegression(learning_rate=0.3, max_iter=1000)
    model.fit(X, y)
    
    preds = model.predict(X)
    acc = np.mean(preds == y)
    
    print(f"\n训练集准确率: {acc:.4f}")
    print(f"类别分布: {Counter(y)}")
    
    # 测试预测
    test_X = np.array([
        [2.5, 2.5],  # 应该预测为 0
        [5.5, 2.5],  # 应该预测为 1
        [3.5, 5.0],  # 应该预测为 2
    ])
    test_preds = model.predict(test_X)
    print(f"\n测试预测:")
    for i, (x, pred) in enumerate(zip(test_X, test_preds)):
        proba = model.predict_proba(x)
        print(f"  {x} -> Class {pred}")
        print(f"    概率分布: {proba}")
    
    return model


if __name__ == "__main__":
    # 创建数据集
    X_binary, y_binary = create_binary_dataset(n_samples=100)
    X_multi, y_multi = create_multinomial_dataset(n_samples=50)
    
    print("数据集信息:")
    print(f"二分类 - 样本数: {len(y_binary)}, 特征数: {X_binary.shape[1]}")
    print(f"  类别分布: {Counter(y_binary)}")
    print(f"多分类 - 样本数: {len(y_multi)}, 特征数: {X_multi.shape[1]}")
    print(f"  类别分布: {Counter(y_multi)}")
    
    # 测试模型
    binary_model = test_binary_logistic(X_binary, y_binary)
    multi_model = test_multinomial_logistic(X_multi, y_multi)

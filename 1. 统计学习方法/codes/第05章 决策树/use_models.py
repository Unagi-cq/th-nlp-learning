"""
test.py

测试三种决策树算法：ID3、C4.5 和 CART

说明：
- 比较三种决策树算法在相同数据集上的表现
- ID3: 基于信息增益
- C4.5: 基于信息增益率
- CART: 基于基尼指数

Date: 2026-02-01
"""

import sys
import os

current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)

import numpy as np
from collections import Counter
from id3 import ID3DecisionTree
from c45 import C45DecisionTree
from cart import CARTDecisionTree


def create_test_dataset(n_samples=10):
    """
    创建测试数据集
    
    参数:
        n_samples (int): 生成的数据样本数量，默认为10
    
    返回:
        X_discrete: 离散特征数据（用于ID3/C4.5）
        X_continuous: 连续特征数据（用于CART）
        y: 标签数据
    """
    # 确保样本数量至少为2
    n_samples = max(2, n_samples)
    
    # 基础模板数据
    # 类别A的离散特征模板（小值特征为主）
    class_a_discrete_templates = [
        [0, 0, 0, 0],  # 小, 小, 小, 小
        [0, 0, 0, 1],  # 小, 小, 小, 大
        [0, 0, 1, 1],  # 小, 小, 大, 大
        [0, 1, 1, 1],  # 小, 大, 大, 大
        [0, 1, 0, 0],  # 小, 大, 小, 小
    ]
    
    # 类别B的离散特征模板（大值特征为主）
    class_b_discrete_templates = [
        [1, 1, 1, 1],  # 大, 大, 大, 大
        [1, 1, 1, 0],  # 大, 大, 大, 小
        [1, 1, 0, 0],  # 大, 大, 小, 小
        [1, 0, 0, 0],  # 大, 小, 小, 小
        [1, 0, 1, 1],  # 大, 小, 大, 大
    ]
    
    # 类别A的连续特征模板（小值特征为主）
    class_a_continuous_templates = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
    ]
    
    # 类别B的连续特征模板（大值特征为主）
    class_b_continuous_templates = [
        [6.7, 3.0, 5.2, 2.3],
        [6.3, 2.5, 5.0, 1.9],
        [6.5, 3.0, 5.2, 2.0],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 5.1, 1.8],
    ]
    
    # 计算每个类别的样本数量（尽量保持平衡）
    n_class_a = n_samples // 2
    n_class_b = n_samples - n_class_a
    
    # 生成离散特征数据
    X_discrete_list = []
    y_list = []
    
    # 生成类别A的样本
    for i in range(n_class_a):
        template_idx = i % len(class_a_discrete_templates)
        # 添加一些随机扰动以增加数据多样性
        sample = class_a_discrete_templates[template_idx].copy()
        if i >= len(class_a_discrete_templates):
            # 对于超出模板数量的样本，添加随机扰动
            for j in range(len(sample)):
                if np.random.random() < 0.3:  # 30%概率改变特征值
                    sample[j] = 1 - sample[j]
        X_discrete_list.append(sample)
        y_list.append('A')
    
    # 生成类别B的样本
    for i in range(n_class_b):
        template_idx = i % len(class_b_discrete_templates)
        sample = class_b_discrete_templates[template_idx].copy()
        if i >= len(class_b_discrete_templates):
            # 对于超出模板数量的样本，添加随机扰动
            for j in range(len(sample)):
                if np.random.random() < 0.3:  # 30%概率改变特征值
                    sample[j] = 1 - sample[j]
        X_discrete_list.append(sample)
        y_list.append('B')
    
    X_discrete = np.array(X_discrete_list)
    y = np.array(y_list)
    
    # 生成连续特征数据
    X_continuous_list = []
    
    # 生成类别A的连续特征样本
    for i in range(n_class_a):
        template_idx = i % len(class_a_continuous_templates)
        sample = class_a_continuous_templates[template_idx].copy()
        if i >= len(class_a_continuous_templates):
            # 对于超出模板数量的样本，添加随机扰动
            noise = np.random.normal(0, 0.1, size=len(sample))
            sample = [s + n for s, n in zip(sample, noise)]
        X_continuous_list.append(sample)
    
    # 生成类别B的连续特征样本
    for i in range(n_class_b):
        template_idx = i % len(class_b_continuous_templates)
        sample = class_b_continuous_templates[template_idx].copy()
        if i >= len(class_b_continuous_templates):
            # 对于超出模板数量的样本，添加随机扰动
            noise = np.random.normal(0, 0.1, size=len(sample))
            sample = [s + n for s, n in zip(sample, noise)]
        X_continuous_list.append(sample)
    
    X_continuous = np.array(X_continuous_list)
    
    return X_discrete, X_continuous, y


def test_id3(X, y):
    """
    测试 ID3 算法
    """
    id3_tree = ID3DecisionTree(min_samples_split=2, max_depth=5)
    id3_tree.fit(X, y)
    
    # 测试预测
    test_X = np.array([
        [0, 0, 0, 0],  # 应该预测为 A
        [1, 1, 1, 1],  # 应该预测为 B
    ])
    
    predictions = id3_tree.predict(test_X)
    
    print("\nID3 预测结果:")
    for x, pred in zip(test_X, predictions):
        print(f"  {x} -> {pred}")
    
    # 打印决策树结构
    print("\n" + "=" * 80)
    print("ID3 决策树结构")
    print("=" * 80)
    id3_tree.print_tree()
    
    return id3_tree


def test_c45(X, y):
    """
    测试 C4.5 算法
    """
    print("\n" + "=" * 80)
    print("测试 C4.5 决策树算法")
    print("=" * 80)
    
    c45_tree = C45DecisionTree(min_samples_split=2, max_depth=5, min_info_gain=1e-6)
    c45_tree.fit(X, y)
    
    # 测试预测
    test_X = np.array([
        [0, 0, 0, 0],  # 应该预测为 A
        [1, 1, 1, 1],  # 应该预测为 B
    ])
    
    predictions = c45_tree.predict(test_X)
    
    print("\nC4.5 预测结果:")
    for x, pred in zip(test_X, predictions):
        print(f"  {x} -> {pred}")
    
    # 打印决策树结构
    print("\n" + "=" * 80)
    print("C4.5 决策树结构")
    print("=" * 80)
    c45_tree.print_tree()
    
    return c45_tree


def test_cart(X, y):
    """
    测试 CART 算法
    """
    print("\n" + "=" * 80)
    print("测试 CART 决策树算法")
    print("=" * 80)
    
    cart_tree = CARTDecisionTree(min_samples_split=2, max_depth=5, min_impurity_decrease=1e-7)
    cart_tree.fit(X, y)
    
    # 测试预测
    test_X = np.array([
        [5.0, 3.5, 1.5, 0.2],  # 应该预测为 A
        [6.5, 3.0, 5.2, 2.0],  # 应该预测为 B
    ])
    
    predictions = cart_tree.predict(test_X)
    
    print("\nCART 预测结果:")
    for x, pred in zip(test_X, predictions):
        print(f"  {x} -> {pred}")
    
    # 打印决策树结构
    print("\n" + "=" * 80)
    print("CART 决策树结构")
    print("=" * 80)
    cart_tree.print_tree()
    
    return cart_tree


if __name__ == "__main__":
    # 配置生成的数据样本数量（可根据需要修改）
    n_samples = 10
    
    # 创建测试数据集
    X_discrete, X_continuous, y = create_test_dataset(n_samples=n_samples)
    
    print("数据集信息:")
    print(f"样本数量: {n_samples}")
    print(f"离散特征形状: {X_discrete.shape}")
    print(f"连续特征形状: {X_continuous.shape}")
    print(f"标签分布: {Counter(y)}")
    print(X_continuous)
    
    # 测试三种算法
    # id3_model = test_id3(X_discrete, y)
    # c45_model = test_c45(X_discrete, y)
    cart_model = test_cart(X_continuous, y)

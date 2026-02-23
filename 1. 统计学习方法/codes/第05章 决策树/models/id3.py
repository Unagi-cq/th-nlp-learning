"""
id3.py

实现 ID3 决策树算法，用于分类任务。

说明：
- ID3 算法基于信息增益进行特征选择
- 信息增益 = 数据集的经验熵 - 特征对数据集的条件熵
- 优先选择信息增益大的特征作为划分标准

Date: 2026-02-01
"""
import numpy as np
from collections import Counter
from typing import Union, List, Tuple, Any


class ID3DecisionTree:
    def __init__(self, min_samples_split: int = 2, max_depth: int = float('inf')):
        """
        初始化 ID3 决策树
        :param min_samples_split: 分割节点所需的最小样本数
        :param max_depth: 树的最大深度
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y: np.ndarray, verbose: bool = False) -> float:
        """
        计算数据集的经验熵 H(D) = -∑(p_i * log2(p_i))
        :param y: 标签数组
        :param verbose: 是否打印详细计算过程
        :return: 经验熵
        """
        if len(y) == 0:
            return 0.0
        
        counts = Counter(y)
        total = len(y)
        entropy = 0.0
        
        if verbose:
            print(f"         │  H(D) = -∑(p_i * log2(p_i))")
            print(f"         │  类别分布: {dict(counts)}, 总样本数: {total}")
            print(f"         │  = -[", end="")
        
        terms = []
        term_values = []
        for label, count in counts.items():
            p = count / total
            if p > 0:
                term = -p * np.log2(p)
                entropy += term
                if verbose:
                    terms.append(f"({count}/{total}) * log2({count}/{total})")
                    term_values.append(f"{term:.4f}")
        
        if verbose:
            print(" + ".join(terms), end="")
            print("]")
            print(f"         │  = -[", end="")
            print(" + ".join(term_values), end="")
            print(f"] = {entropy:.4f}")
        
        return entropy

    def _conditional_entropy(self, X: np.ndarray, y: np.ndarray, feature_idx: int, verbose: bool = False) -> float:
        """
        计算特征对数据集的条件熵 H(D|A)
        :param X: 特征矩阵
        :param y: 标签数组
        :param feature_idx: 特征索引
        :param verbose: 是否打印详细计算过程
        :return: 条件熵
        """
        feature_values = np.unique(X[:, feature_idx])
        cond_entropy = 0.0
        total = len(y)
        
        if verbose:
            print(f"      ┌─ 计算条件熵 H(D|A_{feature_idx}) = ∑(|D_v|/|D| * H(D_v))")
            print(f"      │  特征 A_{feature_idx} 有 {len(feature_values)} 个取值: {sorted(feature_values)}")
            print(f"      │  总样本数 |D| = {total}")
            print(f"      │")
        
        terms = []
        term_values = []
        
        for idx, value in enumerate(feature_values):
            mask = X[:, feature_idx] == value
            subset_y = y[mask]
            subset_size = len(subset_y)
            weight = subset_size / total
            subset_counts = Counter(subset_y)
            
            if verbose:
                print(f"      │  ┌─ 子集 D_{value} (特征 A_{feature_idx} = {value}):")
                print(f"      │  │  样本数 |D_{value}| = {subset_size}/{total} = {weight:.4f}")
                print(f"      │  │  类别分布: {dict(subset_counts)}")
                print(f"      │  │  计算子集经验熵 H(D_{value}):")
            
            subset_entropy = self._entropy(subset_y, verbose=verbose)
            term_value = weight * subset_entropy
            cond_entropy += term_value
            
            if verbose:
                print(f"      │  │  H(D_{value}) = {subset_entropy:.4f}")
                print(f"      │  │  加权项: ({subset_size}/{total}) * {subset_entropy:.4f} = {term_value:.4f}")
                if idx < len(feature_values) - 1:
                    print(f"      │  │")
                terms.append(f"({subset_size}/{total}) * H(D_{value})")
                term_values.append(f"({subset_size}/{total}) * {subset_entropy:.4f}")
        
        if verbose:
            print(f"      │")
            print(f"      │  ┌─ 汇总计算:")
            print(f"      │  │  H(D|A_{feature_idx}) = " + " + ".join(terms))
            print(f"      │  │                  = " + " + ".join(term_values))
            print(f"      │  │                  = {cond_entropy:.4f}")
            print(f"      └─ 条件熵 H(D|A_{feature_idx}) = {cond_entropy:.4f}")
        
        return cond_entropy

    def _information_gain(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
        """
        计算信息增益 g(D,A) = H(D) - H(D|A)
        :param X: 特征矩阵
        :param y: 标签数组
        :param feature_idx: 特征索引
        :return: 信息增益
        """
        base_entropy = self._entropy(y)
        cond_entropy = self._conditional_entropy(X, y, feature_idx)
        return base_entropy - cond_entropy

    def _majority_class(self, y: np.ndarray) -> Any:
        """
        返回样本中的多数类
        :param y: 标签数组
        :return: 多数类标签
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0, used_features: set = None) -> dict:
        """
        递归构建决策树
        :param X: 特征矩阵
        :param y: 标签数组
        :param depth: 当前树的深度
        :param used_features: 已使用的特征索引集合（避免重复使用离散特征）
        :return: 决策树节点
        """
        if used_features is None:
            used_features = set()
        
        # 如果所有样本属于同一类别，返回叶节点
        if len(np.unique(y)) == 1:
            return {'type': 'leaf', 'class': y[0]}
        
        # 如果达到最大深度或样本数不足，返回多数类
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return {'type': 'leaf', 'class': self._majority_class(y)}
        
        # 获取可用特征（排除已使用的特征）
        available_features = [i for i in range(X.shape[1]) if i not in used_features]
        
        # 如果没有可用特征，返回多数类
        if len(available_features) == 0:
            return {'type': 'leaf', 'class': self._majority_class(y)}
        
        # 计算当前数据集的经验熵
        print(f"\n{'='*60}")
        print(f"第 {depth} 层 - 信息增益计算过程")
        print(f"{'='*60}")
        print(f"当前样本数: {len(X)}, 类别分布: {Counter(y)}")
        if used_features:
            print(f"已使用的特征: {sorted(used_features)}")
            print(f"可用特征: {available_features}")
        print(f"\n计算数据集经验熵 H(D):")
        base_entropy = self._entropy(y, verbose=True)
        
        print(f"\n各特征的信息增益计算:")
        
        # 寻找最优分割特征（只考虑可用特征）
        best_feature_idx = -1
        best_gain = -1
        feature_gains = []
        
        for i in available_features:
            print(f"\n  特征 {i}:")
            # 计算条件熵
            cond_entropy = self._conditional_entropy(X, y, i, verbose=True)
            # 计算信息增益
            gain = self._information_gain(X, y, i)
            feature_gains.append((i, gain, cond_entropy))
            
            print(f"    - 信息增益 g(D,A_{i}) = H(D) - H(D|A_{i}) = {base_entropy:.4f} - {cond_entropy:.4f} = {gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = i
        
        # 如果没有找到有效分割特征，返回多数类
        if best_feature_idx == -1 or best_gain <= 0:
            return {'type': 'leaf', 'class': self._majority_class(y)}
        
        print(f"\n最优特征: 特征 {best_feature_idx}, 信息增益: {best_gain:.4f}")
        
        # 创建内部节点
        node = {
            'type': 'internal',
            'feature_idx': best_feature_idx,
            'children': {}
        }
        
        # 将当前特征添加到已使用特征集合
        new_used_features = used_features.copy()
        new_used_features.add(best_feature_idx)
        
        # 获取最优特征的所有取值
        feature_values = np.unique(X[:, best_feature_idx])
        
        for value in feature_values:
            # 分割数据集
            mask = X[:, best_feature_idx] == value
            X_subset = X[mask]
            y_subset = y[mask]
            
            # 递归构建子树（传递已使用的特征集合）
            child_node = self._build_tree(X_subset, y_subset, depth + 1, new_used_features)
            node['children'][value] = child_node
        
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ID3DecisionTree':
        """
        训练 ID3 决策树
        :param X: 训练特征矩阵
        :param y: 训练标签数组
        :return: 自身
        """
        print("=" * 60)
        print("开始训练 ID3 决策树")
        print("=" * 60)
        print(f"训练样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"类别分布: {Counter(y)}")
        
        self.tree = self._build_tree(X, y)
        print("\nID3 决策树构建完成！")
        return self

    def _predict_sample(self, sample: np.ndarray, node: dict) -> Any:
        """
        预测单个样本
        :param sample: 单个样本特征
        :param node: 当前节点
        :return: 预测类别
        """
        if node['type'] == 'leaf':
            return node['class']
        
        feature_val = sample[node['feature_idx']]
        if feature_val in node['children']:
            return self._predict_sample(sample, node['children'][feature_val])
        else:
            # 如果特征值在训练中未见过，返回该节点最常见的类别
            return node['children'][list(node['children'].keys())[0]]['class']

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测多个样本
        :param X: 测试特征矩阵
        :return: 预测标签数组
        """
        predictions = []
        for sample in X:
            pred = self._predict_sample(sample, self.tree)
            predictions.append(pred)
        return np.array(predictions)

    def print_tree(self, node: dict = None, indent: str = "", prefix: str = ""):
        """
        打印决策树结构
        :param node: 当前节点，默认为根节点
        :param indent: 当前缩进
        :param prefix: 前缀标识
        """
        if node is None:
            node = self.tree
        
        if node['type'] == 'leaf':
            print(f"{indent}{prefix}类别: {node['class']}")
        else:
            print(f"{indent}{prefix}特征 {node['feature_idx']}")
            children = node['children']
            for i, (value, child) in enumerate(children.items()):
                is_last = (i == len(children) - 1)
                branch = "└── " if is_last else "├── "
                next_indent = indent + ("    " if is_last else "│   ")
                print(f"{indent}{branch}值={value}:")
                self.print_tree(child, next_indent, "")

"""
cart.py

实现 CART 决策树算法（分类回归树），用于分类任务。

说明：
- CART 算法基于基尼指数进行特征选择
- 基尼指数表示集合的不确定性，基尼指数越大，样本的不确定性越高
- CART 是二叉树，每次分割将数据分为两部分

Date: 2026-02-01
"""
import numpy as np
from collections import Counter
from typing import Union, List, Tuple, Any


class CARTDecisionTree:
    def __init__(self, min_samples_split: int = 2, max_depth: int = float('inf'), 
                 min_impurity_decrease: float = 1e-7):
        """
        初始化 CART 决策树
        :param min_samples_split: 分割节点所需的最小样本数
        :param max_depth: 树的最大深度
        :param min_impurity_decrease: 最小纯度减少阈值，低于此值则不分割
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def _gini(self, y: np.ndarray, verbose: bool = False) -> float:
        """
        计算数据集的基尼指数 Gini(D) = 1 - ∑(p_i)^2
        :param y: 标签数组
        :param verbose: 是否打印详细计算过程
        :return: 基尼指数
        """
        if len(y) == 0:
            return 0.0
        
        counts = Counter(y)
        total = len(y)
        gini = 1.0
        
        if verbose:
            print(f"      Gini(D) = 1 - ∑(p_i)^2")
            print(f"      = 1 - [", end="")
        
        terms = []
        term_values = []
        
        for label, count in counts.items():
            p = count / total
            p_squared = p * p
            gini -= p_squared
            if verbose:
                terms.append(f"({count}/{total})^2")
                term_values.append(f"{p_squared:.4f}")
        
        if verbose:
            print(" + ".join(terms), end="")
            print("]")
            print(f"      = 1 - [", end="")
            print(" + ".join(term_values), end="")
            sum_p_squared = sum(float(v) for v in term_values)
            print(f"] = 1 - {sum_p_squared:.4f} = {gini:.4f}")
        
        return gini

    def _gini_split(self, X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
        """
        计算按某个特征和阈值分割后的基尼指数
        :param X: 特征矩阵
        :param y: 标签数组
        :param feature_idx: 特征索引
        :param threshold: 分割阈值
        :return: 加权基尼指数
        """
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        n_left, n_right = len(left_y), len(right_y)
        n_total = len(y)
        
        if n_left == 0 or n_right == 0:
            return float('inf')  # 无效分割
        
        weighted_gini = (n_left / n_total) * self._gini(left_y) + \
                       (n_right / n_total) * self._gini(right_y)
        
        return weighted_gini

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, depth: int = 0, verbose: bool = True) -> Tuple[int, float, float]:
        """
        寻找最佳分割点
        :param X: 特征矩阵
        :param y: 标签数组
        :param depth: 当前深度（用于打印）
        :param verbose: 是否打印详细信息
        :return: (最佳特征索引, 最佳阈值, 最佳基尼指数)
        """
        best_feature_idx = -1
        best_threshold = 0.0
        best_gini = float('inf')
        
        n_features = X.shape[1]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"第 {depth} 层 - 基尼指数计算过程")
            print(f"{'='*60}")
            print(f"当前样本数: {len(X)}, 类别分布: {Counter(y)}")
            print(f"\n计算当前数据集基尼指数 Gini(D):")
            current_gini = self._gini(y, verbose=True)
            print(f"\n各特征和阈值的基尼指数计算:")
        else:
            current_gini = self._gini(y)
        
        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值并排序
            unique_values = np.unique(X[:, feature_idx])
            
            if verbose and len(unique_values) > 1:
                print(f"\n  特征 {feature_idx} (取值范围: [{unique_values.min():.2f}, {unique_values.max():.2f}]):")
            
            # 尝试在相邻值的中点进行分割
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
                gini = self._gini_split(X, y, feature_idx, threshold)
                
                if verbose:
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask
                    left_y = y[left_mask]
                    right_y = y[right_mask]
                    n_left, n_right = len(left_y), len(right_y)
                    n_total = len(y)
                    
                    print(f"\n    阈值 = {threshold:.4f}:")
                    print(f"      左子集: {n_left} 个样本, 类别分布 = {Counter(left_y)}")
                    print(f"      计算左子集基尼指数:")
                    left_gini = self._gini(left_y, verbose=True)
                    print(f"      右子集: {n_right} 个样本, 类别分布 = {Counter(right_y)}")
                    print(f"      计算右子集基尼指数:")
                    right_gini = self._gini(right_y, verbose=True)
                    print(f"      加权基尼指数 = ({n_left}/{n_total}) * {left_gini:.4f} + ({n_right}/{n_total}) * {right_gini:.4f} = {gini:.4f}")
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        if verbose:
            print(f"\n最优分割: 特征 {best_feature_idx}, 阈值 = {best_threshold:.4f}, 基尼指数 = {best_gini:.4f}")
            print(f"基尼指数减少量 = {current_gini:.4f} - {best_gini:.4f} = {current_gini - best_gini:.4f}")
        
        return best_feature_idx, best_threshold, best_gini

    def _majority_class(self, y: np.ndarray) -> Any:
        """
        返回样本中的多数类
        :param y: 标签数组
        :return: 多数类标签
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """
        递归构建决策树
        :param X: 特征矩阵
        :param y: 标签数组
        :param depth: 当前树的深度
        :return: 决策树节点
        """
        # 如果所有样本属于同一类别，返回叶节点
        if len(np.unique(y)) == 1:
            return {'type': 'leaf', 'class': y[0]}
        
        # 如果达到最大深度或样本数不足，返回多数类
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return {'type': 'leaf', 'class': self._majority_class(y)}
        
        # 寻找最佳分割
        best_feature_idx, best_threshold, best_gini = self._find_best_split(X, y, depth)
        
        # 如果无法找到有效分割或基尼指数改善不够，返回多数类
        current_gini = self._gini(y)
        impurity_decrease = current_gini - best_gini
        
        if best_feature_idx == -1 or impurity_decrease < self.min_impurity_decrease:
            return {'type': 'leaf', 'class': self._majority_class(y)}
        
        # 创建内部节点
        node = {
            'type': 'internal',
            'feature_idx': best_feature_idx,
            'threshold': best_threshold,
            'impurity': current_gini,
            'children': {}
        }
        
        # 分割数据集
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # 递归构建左右子树
        node['children']['left'] = self._build_tree(X_left, y_left, depth + 1)
        node['children']['right'] = self._build_tree(X_right, y_right, depth + 1)
        
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CARTDecisionTree':
        """
        训练 CART 决策树
        :param X: 训练特征矩阵
        :param y: 训练标签数组
        :return: 自身
        """
        print("=" * 60)
        print("开始训练 CART 决策树")
        print("=" * 60)
        print(f"训练样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"类别分布: {Counter(y)}")
        print(f"初始基尼指数: {self._gini(y):.4f}")
        
        self.tree = self._build_tree(X, y)
        print("\nCART 决策树构建完成！")
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
        if feature_val <= node['threshold']:
            return self._predict_sample(sample, node['children']['left'])
        else:
            return self._predict_sample(sample, node['children']['right'])

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
            print(f"{indent}{prefix}特征 {node['feature_idx']} <= {node['threshold']:.4f} (基尼指数: {node.get('impurity', 0):.4f})")
            children = node['children']
            # CART是二叉树，只有left和right
            if 'left' in children:
                print(f"{indent}├── 是:")
                self.print_tree(children['left'], indent + "│   ", "")
            if 'right' in children:
                print(f"{indent}└── 否:")
                self.print_tree(children['right'], indent + "    ", "")

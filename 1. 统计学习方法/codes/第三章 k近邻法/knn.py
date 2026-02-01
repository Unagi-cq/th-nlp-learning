"""
knn.py

实现 KD-Tree 的 1-NN（最近邻）分类，用于测试 KD-Tree 在不同维度下的性能。

说明：
- KD-Tree 通过空间分割减少搜索范围，候选点比较次数会显著低于暴力搜索
- 在低维时效果明显，高维时可能退化（维度灾难）

Author: Huang CQ
Date: 2026-01-26
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from dataclasses import dataclass


@dataclass
class SearchStats:
    """记录一次 knn 查询的统计量。"""

    # 候选点的比较次数（KD-Tree 会远小于 n_train）
    candidate_comparisons: int = 0


@dataclass
class KDNode:
    """KD-Tree 节点（仅用于 L2 距离的 1-NN 查询）。"""

    axis: int
    index: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree1NN:
    """
    KD-Tree 实现的 1-NN（最近邻）分类器，使用 L2 平方距离。

    说明：
    - 通过构建 KD-Tree 减少搜索范围，在低维时效果显著
    - 高维时可能退化，候选点比较次数接近暴力搜索
    """

    def __init__(self):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.root: Optional[KDNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KDTree1NN":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y 必须是一维数组且长度与 X 的样本数一致")

        self.X = X
        self.y = y
        idxs = np.arange(X.shape[0], dtype=int)
        self.root = self._build(idxs, depth=0)
        return self

    def _build(self, idxs: np.ndarray, depth: int) -> Optional[KDNode]:
        assert self.X is not None
        if idxs.size == 0:
            return None
        axis = depth % self.X.shape[1]

        # 用 axis 维排序，取中位数，保证树尽量平衡
        order = np.argsort(self.X[idxs, axis], kind="mergesort")
        idxs = idxs[order]
        mid = idxs.size // 2
        node = KDNode(axis=axis, index=int(idxs[mid]))
        node.left = self._build(idxs[:mid], depth + 1)
        node.right = self._build(idxs[mid + 1:], depth + 1)
        return node

    @staticmethod
    def _l2_sq_point(x: np.ndarray, xi: np.ndarray) -> float:
        """计算 L2 平方距离。"""
        diff = x - xi
        return float(np.dot(diff, diff))

    def kneighbor(self, x: np.ndarray) -> Tuple[int, float, SearchStats]:
        """
        返回 (最近邻索引, L2平方距离, stats)。
        """
        x = np.asarray(x, dtype=float).reshape(-1)

        stats = SearchStats(candidate_comparisons=0)
        best_idx = self.root.index
        best = float("inf")

        def search(node: Optional[KDNode]):
            nonlocal best, best_idx
            if node is None:
                return

            xi = self.X[node.index]
            stats.candidate_comparisons += 1
            dist = self._l2_sq_point(x, xi)
            if dist < best:
                best = dist
                best_idx = node.index

            axis = node.axis
            diff = x[axis] - xi[axis]
            first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)

            # 先走更可能包含近邻的子树
            search(first)

            # 再判断是否需要回溯：分割超平面到 x 的距离若已超过 best，则可剪枝
            if diff * diff < best:
                search(second)

        search(self.root)
        return int(best_idx), float(best), stats

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, SearchStats]:
        X = np.asarray(X, dtype=float)

        preds = np.empty(X.shape[0], dtype=self.y.dtype)
        agg = SearchStats(0)
        for i in range(X.shape[0]):
            idx, _, st = self.kneighbor(X[i])
            preds[i] = self.y[idx]
            agg.candidate_comparisons += st.candidate_comparisons
        return preds, agg


def make_two_cluster_data(
        n_train: int,
        n_query: int,
        dim: int,
        seed: int = 0,
        center_sep: float = 2.5,
        noise: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成两团高斯数据（用于分类），方便观察不同 metric/维度下的差异。
    """
    rng = np.random.default_rng(seed)
    c0 = -center_sep / 2.0
    c1 = +center_sep / 2.0
    X0 = rng.normal(loc=c0, scale=noise, size=(n_train // 2, dim))
    X1 = rng.normal(loc=c1, scale=noise, size=(n_train - n_train // 2, dim))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

    Q0 = rng.normal(loc=c0, scale=noise, size=(n_query // 2, dim))
    Q1 = rng.normal(loc=c1, scale=noise, size=(n_query - n_query // 2, dim))
    X_query = np.vstack([Q0, Q1])
    y_query = np.array([0] * Q0.shape[0] + [1] * Q1.shape[0])

    # 打乱
    p_tr = rng.permutation(n_train)
    p_q = rng.permutation(n_query)
    return X_train[p_tr], y_train[p_tr], X_query[p_q], y_query[p_q]


def run_experiment(
        dims: List[int],
        n_train: int = 2000,
        n_query: int = 200,
        seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """
    对不同维度测试 KD-Tree 的性能，返回每组的统计结果：
    - acc: 分类准确率
    - avg_cc: 每个查询平均比较的候选点数（candidate comparisons）
    """
    results: Dict[int, Dict[str, float]] = {}
    for dim in dims:
        Xtr, ytr, Xq, yq = make_two_cluster_data(
            n_train=n_train, n_query=n_query, dim=dim, seed=seed, center_sep=2.5, noise=1.0
        )
        # 使用 KD-Tree 进行 1-NN 分类
        model = KDTree1NN().fit(Xtr, ytr)
        pred, st = model.predict(Xq)

        acc = float(np.mean(pred == yq))
        avg_cc = st.candidate_comparisons / n_query

        results[dim] = {
            "acc": acc,
            "avg_cc": float(avg_cc),
        }
    return results


# ------------------ 测试 KD-Tree 在不同维度下的性能 ------------------
if __name__ == "__main__":
    dims = [2, 4, 8, 16, 32, 64]
    n_train = 3000
    n_query = 100
    seed = 0

    print("===== KD-Tree 1-NN 测试 =====")
    print(f"训练样本数: {n_train}, 查询样本数: {n_query}\n")

    results = run_experiment(
        dims=dims,
        n_train=n_train,
        n_query=n_query,
        seed=seed,
    )

    print("结果汇总：")
    print("-" * 90)
    for d in dims:
        r = results[d]
        print(
            f"dim={d:>3} | "
            f"acc={r['acc']:.3f} | "
            f"avg_cc={r['avg_cc']:.1f}"
        )
    print("-" * 90)

    print("\n测试完成！")

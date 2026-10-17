"""
svm_linear_sklearn.py

使用sklearn实现简单的线性可分SVM（原始形式），用于二分类任务。

Date: 2026-03-21
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class LinearSVMsklearn:
    def __init__(self):
        self.w = None
        self.b = 0
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.svc = None

    def fit(self, X, y):
        """
        使用sklearn SVC求解线性可分SVM
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的标签，取值为 +1 或 -1
        """
        self.svc = SVC(kernel='linear', C=1e5)
        self.svc.fit(X, y)

        # 支持向量
        self.support_vectors = self.svc.support_vectors_
        support_vector_indices = self.svc.support_
        self.support_vector_labels = y[support_vector_indices]

        # sklearn不直接提供alpha，通过支持向量重构
        # 对于线性SVM，alpha_i * y_i = sum_j alpha_j * y_j * x_j^T * x_i
        # 简化处理：使用coef_和support_vectors估算
        self.w = self.svc.coef_[0]
        self.b = self.svc.intercept_[0]

        # 估算alpha（仅用于展示）
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        for i, idx in enumerate(support_vector_indices):
            self.alphas[idx] = 1.0 / np.linalg.norm(self.w)

        print(f"支持向量个数: {len(self.support_vectors)}/{n_samples}")
        print(f"权重 w: {self.w}")
        print(f"偏置 b: {self.b}")

    def predict(self, X):
        return self.svc.predict(X)

    def score(self, X, y):
        return self.svc.score(X, y)


# ------------------ 示例：二维线性可分数据 ------------------
if __name__ == "__main__":
    # 构造简单线性可分数据（来自李航《统计学习方法》例7.1）
    X = np.array([
        [3, 3],
        [4, 3],
        [0, 1]
    ])
    y = np.array([1, 1, -1])

    # 训练SVM
    svm = LinearSVMsklearn()
    svm.fit(X, y)

    # 预测
    preds = svm.predict(X)
    print("\n预测结果:", preds)
    print("真实标签:", y)
    print("准确率:", svm.score(X, y))

    # 打印支持向量
    print(f"\n支持向量: {svm.support_vectors}")
    print(f"支持向量标签: {svm.support_vector_labels}")

    # 可视化（仅限二维）
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1', s=100)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1', s=100)

    # 标记支持向量
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                s=200, facecolors='none', edgecolors='green', linewidth=2,
                label='Support Vectors')

    # 绘制决策边界
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x0_vals = np.linspace(x0_min, x0_max, 100)
    x1_vals = -(svm.w[0] * x0_vals + svm.b) / svm.w[1]
    plt.plot(x0_vals, x1_vals, 'k-', label='Decision Boundary')

    # 绘制间隔边界
    x1_upper = -(svm.w[0] * x0_vals + svm.b - 1) / svm.w[1]
    x1_lower = -(svm.w[0] * x0_vals + svm.b + 1) / svm.w[1]
    plt.plot(x0_vals, x1_upper, 'k--', alpha=0.5)
    plt.plot(x0_vals, x1_lower, 'k--', alpha=0.5)

    plt.legend()
    plt.title("Linear SVM (sklearn) Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

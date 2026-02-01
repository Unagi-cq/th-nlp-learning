"""
test_perceptron.py

实现感知机算法（原始形式），用于二分类任务。

Author: Huang CQ
Date: 2026-01-16
"""
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=1.0, max_iter=1000):
        self.eta = learning_rate  # 学习率 η
        self.max_iter = max_iter  # 最大迭代次数
        self.w = None  # 权重向量 w
        self.b = 0  # 偏置 b

    def fit(self, X, y):
        """
        训练感知机模型
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的标签，取值为 +1 或 -1
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化 w = 0
        self.b = 0

        print(f"初始化权重: {self.w}, 初始化偏置: {self.b}")
        count = 0
        for epoch in range(self.max_iter):
            errors = 0
            for i in range(n_samples):
                count += 1
                xi, yi = X[i], y[i]
                print(f"epoch: {epoch + 1}, count: {count}, 点({xi}, {yi}) 误分类= {yi * (np.dot(self.w, xi) + self.b) <= 0}, w: {self.w}, b: {self.b}, w*xi+b: {np.dot(self.w, xi) + self.b}")
                # 判断是否误分类
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    # 更新
                    self.w += self.eta * yi * xi
                    self.b += self.eta * yi
                    errors += 1
            print("\n")
            if errors == 0:
                print(f"在第 {count} 次迭代后，误分类点全部被正确分类")
                break
        else:
            print("达到最大迭代次数，模型可能未收敛")

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


# ------------------ 示例：二维线性可分数据 ------------------
if __name__ == "__main__":
    # 构造简单线性可分数据
    X = np.array([
        [3, 3],
        [4, 3],
        [1, 1]
    ])
    y = np.array([1, 1, -1])*0.1

    # 训练感知机
    perceptron = Perceptron(learning_rate=1.0, max_iter=1000)
    perceptron.fit(X, y)

    # 预测
    preds = perceptron.predict(X)
    print("预测结果:", preds)
    print("真实标签:", y)
    print("权重:", perceptron.w)
    print("偏置:", perceptron.b)

    # 可视化（仅限二维）
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')

    # 绘制决策边界: w0*x0 + w1*x1 + b = 0 → x1 = -(w0*x0 + b)/w1
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x0_vals = np.linspace(x0_min, x0_max, 100)
    x1_vals = -(perceptron.w[0] * x0_vals + perceptron.b) / perceptron.w[1]
    plt.plot(x0_vals, x1_vals, 'k--', label='Decision Boundary')

    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

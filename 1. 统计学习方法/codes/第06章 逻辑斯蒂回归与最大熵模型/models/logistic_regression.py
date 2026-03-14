"""
logistic_regression.py

实现逻辑斯蒂回归（Logistic Regression），用于二分类和多分类任务。
- 二分类：BinaryLogisticRegression
- 多分类：MultinomialLogisticRegression

Date: 2026-03-14
"""

import numpy as np


class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        """
        初始化二项逻辑回归模型
        :param learning_rate: 学习率
        :param max_iter: 最大迭代次数
        """
        self.eta = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        """
        Sigmoid函数
        :param z: 线性组合
        :return: sigmoid值
        """
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        训练二项逻辑回归模型
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的标签，取值为 0 或 1
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        print(f"初始化权重: {self.w}, 初始化偏置: {self.b}")

        for epoch in range(1, self.max_iter + 1):
            linear = np.dot(X, self.w) + self.b
            y_prob = self._sigmoid(linear)

            # 负对数似然损失
            eps = 1e-12
            loss = -np.mean(y * np.log(y_prob + eps) + (1 - y) * np.log(1 - y_prob + eps))

            # 梯度
            grad_w = np.dot(X.T, (y_prob - y)) / n_samples
            grad_b = np.mean(y_prob - y)

            # 参数更新
            self.w -= self.eta * grad_w
            self.b -= self.eta * grad_b

            if epoch % 100 == 0 or epoch == 1:
                print(
                    f"epoch: {epoch}, loss: {loss:.4f}, "
                    f"w: {self.w}, b: {self.b}"
                )

    def predict_proba(self, X):
        """
        预测样本属于正类的概率
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples,) 概率值
        """
        linear = np.dot(X, self.w) + self.b
        return self._sigmoid(linear)

    def predict(self, X, threshold=0.5):
        """
        预测样本类别
        :param X: (n_samples, n_features) 的输入数据
        :param threshold: 分类阈值
        :return: (n_samples,) 预测类别
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        """
        初始化多项逻辑回归模型（Softmax回归）
        :param learning_rate: 学习率
        :param max_iter: 最大迭代次数
        """
        self.eta = learning_rate
        self.max_iter = max_iter
        self.W = None
        self.b = None

    def _softmax(self, Z):
        """
        Softmax函数
        :param Z: (n_samples, n_classes) 线性组合
        :return: (n_samples, n_classes) 概率分布
        """
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        训练多项逻辑回归模型
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的标签，取值为 0,1,...,K-1
        """
        n_samples, n_features = X.shape
        n_classes = int(np.max(y)) + 1

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        print(f"初始化权重矩阵 W 形状: {self.W.shape}, 初始化偏置 b 形状: {self.b.shape}")

        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y] = 1

        for epoch in range(1, self.max_iter + 1):
            Z = np.dot(X, self.W) + self.b
            y_prob = self._softmax(Z)

            eps = 1e-12
            loss = -np.mean(np.sum(Y_onehot * np.log(y_prob + eps), axis=1))

            grad_W = np.dot(X.T, (y_prob - Y_onehot)) / n_samples
            grad_b = np.mean(y_prob - Y_onehot, axis=0)

            self.W -= self.eta * grad_W
            self.b -= self.eta * grad_b

            if epoch % 100 == 0 or epoch == 1:
                print(f"epoch: {epoch}, loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        预测样本属于各类别的概率
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples, n_classes) 概率分布
        """
        X = np.atleast_2d(X)  # 确保 X 是二维数组
        Z = np.dot(X, self.W) + self.b
        return self._softmax(Z)

    def predict(self, X):
        """
        预测样本类别
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples,) 预测类别
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

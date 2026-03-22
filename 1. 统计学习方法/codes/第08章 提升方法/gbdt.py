"""
gbdt.py

实现GBDT（梯度提升决策树）算法，用于回归和分类任务。
根据李航《统计学习方法》第8章实现。

Date: 2026-03-22
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class GBDTRegression:
    """
    GBDT回归模型
    使用平方损失函数
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        :param n_estimators: 弱学习器数量
        :param learning_rate: 学习率
        :param max_depth: 决策树最大深度
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = 0
    
    def fit(self, X, y):
        """
        训练GBDT回归模型
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的目标值
        """
        n_samples = X.shape[0]
        
        # 初始化预测值为均值
        self.initial_prediction = np.mean(y)
        F = np.full(n_samples, self.initial_prediction)
        
        for i in range(self.n_estimators):
            # 计算负梯度（残差）
            residuals = y - F
            
            # 拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            
            # 更新预测值
            F += self.learning_rate * tree.predict(X)
            
            self.trees.append(tree)
            
            # 计算MSE
            mse = np.mean((y - F) ** 2)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"迭代 {i+1}/{self.n_estimators}: MSE={mse:.4f}")
    
    def predict(self, X):
        """
        预测
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples,) 预测值
        """
        F = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        return F


class GBDTClassifier:
    """
    GBDT二分类模型
    使用对数损失函数（log loss）
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        :param n_estimators: 弱学习器数量
        :param learning_rate: 学习率
        :param max_depth: 决策树最大深度
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = 0
    
    def fit(self, X, y):
        """
        训练GBDT分类模型
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的标签，取值为 +1 或 -1
        """
        n_samples = X.shape[0]
        
        # 初始化预测值为0
        self.initial_prediction = 0
        F = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # 计算负梯度（对数损失的梯度）
            # L = log(1 + exp(-y * F)), 梯度 = -y / (1 + exp(y * F))
            residuals = -y / (1 + np.exp(y * F))
            
            # 拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            
            # 更新预测值
            F += self.learning_rate * tree.predict(X)
            
            self.trees.append(tree)
            
            # 计算分类错误率
            probs = 1 / (1 + np.exp(-F))
            preds = np.where(probs > 0.5, 1, -1)
            error = np.mean(preds != y)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"迭代 {i+1}/{self.n_estimators}: 错误率={error:.4f}")
    
    def predict(self, X):
        """
        预测类别
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples,) 预测类别
        """
        F = np.zeros(X.shape[0])
        
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        probs = 1 / (1 + np.exp(-F))
        return np.where(probs > 0.5, 1, -1)
    
    def score(self, X, y):
        """
        计算准确率
        """
        preds = self.predict(X)
        return np.mean(preds == y)


# ------------------ 回归示例 ------------------
def generate_regression_data():
    """生成回归数据"""
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y = X ** 2 + 2 * X + np.random.randn(100) * 0.5
    return X.reshape(-1, 1), y


def demo_regression():
    print("=" * 60)
    print("GBDT回归示例")
    print("=" * 60)
    
    X, y = generate_regression_data()
    
    # 训练GBDT
    gbdt = GBDTRegression(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbdt.fit(X, y)
    
    # 预测
    y_pred = gbdt.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f"\n最终MSE: {mse:.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 子图1：拟合效果
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, c='blue', label='真实值', alpha=0.6)
    plt.plot(X, y_pred, 'r-', linewidth=2, label='GBDT预测')
    plt.title("GBDT回归拟合效果")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    
    # 子图2：残差图
    plt.subplot(1, 2, 2)
    residuals = y - y_pred
    plt.scatter(X, residuals, c='green', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title("残差图")
    plt.xlabel("X")
    plt.ylabel("残差")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("gbdt_regression_demo.png", dpi=300, bbox_inches='tight')
    plt.show()


# ------------------ 分类示例 ------------------
def generate_classification_data():
    """生成分类数据"""
    np.random.seed(42)
    
    # 正类
    pos1 = np.random.randn(40, 2) + np.array([2, 2])
    pos2 = np.random.randn(20, 2) + np.array([-2, -2])
    
    # 负类
    neg = np.random.randn(60, 2) + np.array([0, 0])
    
    X = np.vstack([pos1, pos2, neg])
    y = np.hstack([np.ones(60), -np.ones(60)])
    
    return X, y


def demo_classification():
    print("\n" + "=" * 60)
    print("GBDT二分类示例")
    print("=" * 60)
    
    X, y = generate_classification_data()
    
    # 训练GBDT
    gbdt = GBDTClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbdt.fit(X, y)
    
    # 预测
    preds = gbdt.predict(X)
    accuracy = gbdt.score(X, y)
    print(f"\n训练集准确率: {accuracy:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = gbdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class +1', 
                edgecolors='k', s=50)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1', 
                edgecolors='k', s=50)
    
    plt.title(f"GBDT分类决策边界 (准确率: {accuracy:.4f})")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("gbdt_classification_demo.png", dpi=300, bbox_inches='tight')
    plt.show()


# ------------------ 对比GBDT与GBDT ------------------
def demo_comparison():
    print("\n" + "=" * 60)
    print("GBDT与AdaBoost对比")
    print("=" * 60)
    
    X, y = generate_classification_data()
    
    from adaboost import AdaBoost
    
    # AdaBoost
    print("\n1. AdaBoost")
    ada = AdaBoost(n_estimators=50, learning_rate=1.0)
    ada.fit(X, y)
    print(f"   准确率: {ada.score(X, y):.4f}")
    
    # GBDT
    print("\n2. GBDT")
    gbdt = GBDTClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    gbdt.fit(X, y)
    print(f"   准确率: {gbdt.score(X, y):.4f}")


if __name__ == "__main__":
    demo_regression()
    demo_classification()
    demo_comparison()

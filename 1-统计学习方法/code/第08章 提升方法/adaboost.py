"""
adaboost.py

实现AdaBoost算法（自适应提升方法），用于二分类任务。

Date: 2026-03-22
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.tree import DecisionTreeClassifier

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        AdaBoost分类器
        :param n_estimators: 弱学习器数量
        :param learning_rate: 学习率，缩小每轮的权重更新
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []
        self.trainer = DecisionTreeClassifier(max_depth=2)

    def fit(self, X, y):
        """
        训练AdaBoost模型
        :param X: (n_samples, n_features) 的输入数据
        :param y: (n_samples,) 的标签，取值为 +1 或 -1
        """
        n_samples = X.shape[0]
        
        # 初始化样本权重，均匀分布
        weights = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            # 训练弱学习器
            model = self.trainer.fit(X, y, sample_weight=weights)
            
            # 预测
            preds = model.predict(X)
            
            # 计算加权错误率
            miss = (preds != y).astype(int)
            error = np.dot(weights, miss) / np.sum(weights)
            
            # 计算弱学习器权重
            # alpha = 1/2 * ln((1 - error) / error)
            if error == 0:
                alpha = 10.0  # 防止除零
            else:
                alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # 更新样本权重
            # weights = weights * exp(-alpha * y * preds)
            weights = weights * np.exp(-alpha * y * preds)
            weights = weights / np.sum(weights)  # 归一化
            
            self.models.append(model)
            self.model_weights.append(alpha)
            
            print(f"迭代 {i+1}/{self.n_estimators}: 错误率={error:.4f}, 权重α={alpha:.4f}")
            
            # 如果错误率接近0，提前结束
            if error < 1e-10:
                break
        
        print(f"\n训练完成，共使用 {len(self.models)} 个弱学习器")

    def predict(self, X):
        """
        预测类别
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples,) 预测的类别标签
        """
        # 加权投票
        result = np.zeros(X.shape[0])
        
        for model, alpha in zip(self.models, self.model_weights):
            preds = model.predict(X)
            result += alpha * preds
        
        return np.sign(result)

    def predict_proba(self, X):
        """
        预测概率（用于评估）
        :param X: (n_samples, n_features) 的输入数据
        :return: (n_samples,) 预测的置信度
        """
        result = np.zeros(X.shape[0])
        
        for model, alpha in zip(self.models, self.model_weights):
            preds = model.predict(X)
            result += alpha * preds
        
        return result

    def score(self, X, y):
        """
        计算准确率
        :param X: 输入数据
        :param y: 真实标签
        :return: 准确率
        """
        preds = self.predict(X)
        return np.mean(preds == y)


# ------------------ 示例1：简单二维数据 ------------------
def generate_simple_data():
    """生成简单二维分类数据"""
    np.random.seed(42)
    
    # 正类：两个高斯分布
    pos1 = np.random.randn(30, 2) + np.array([2, 2])
    pos2 = np.random.randn(20, 2) + np.array([-2, -2])
    
    # 负类：一个高斯分布
    neg = np.random.randn(50, 2) + np.array([0, 0])
    
    X = np.vstack([pos1, pos2, neg])
    y = np.hstack([np.ones(50), -np.ones(50)])
    
    return X, y


def demo_simple_data():
    print("=" * 60)
    print("示例1：简单二维数据")
    print("=" * 60)
    
    X, y = generate_simple_data()
    
    # 训练AdaBoost
    ada = AdaBoost(n_estimators=500, learning_rate=1.0)
    ada.fit(X, y)
    
    # 预测
    preds = ada.predict(X)
    print(f"\n训练集准确率: {ada.score(X, y):.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 子图1：原始数据
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class +1', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1', alpha=0.6)
    plt.title("原始数据")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.grid(True)
    
    # 子图2：决策边界
    plt.subplot(1, 2, 2)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class +1', edgecolors='k')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1', edgecolors='k')
    plt.title(f"AdaBoost 决策边界 (准确率: {ada.score(X, y):.4f})")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("adaboost_simple_demo.png", dpi=300, bbox_inches='tight')
    plt.show()


# ------------------ 示例2：对比不同弱学习器 ------------------
def demo_weak_learners():
    print("\n" + "=" * 60)
    print("示例2：不同弱学习器的对比")
    print("=" * 60)
    
    X, y = generate_simple_data()
    
    from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
    
    # AdaBoost with Decision Tree (深度为1)
    print("\n1. 决策树桩 (决策树深度=1)")
    ada_tree = AdaBoost(n_estimators=50, learning_rate=1.0)
    ada_tree.fit(X, y)
    print(f"   训练准确率: {ada_tree.score(X, y):.4f}")
    
    # sklearn的AdaBoost
    print("\n2. sklearn AdaBoost (决策树桩)")
    sklearn_ada = SklearnAdaBoost(n_estimators=50, random_state=42)
    sklearn_ada.fit(X, y)
    print(f"   训练准确率: {sklearn_ada.score(X, y):.4f}")
    
    # AdaBoost with different learning rates
    print("\n3. 不同学习率对比")
    learning_rates = [0.5, 1.0, 2.0]
    
    for lr in learning_rates:
        ada_lr = AdaBoost(n_estimators=50, learning_rate=lr)
        ada_lr.fit(X, y)
        print(f"   学习率={lr}: 训练准确率={ada_lr.score(X, y):.4f}, "
              f"使用{len(ada_lr.models)}个弱学习器")


# ------------------ 示例3：非线性可分数据 ------------------
def generate_xor_data():
    """生成XOR数据（非线性可分）"""
    np.random.seed(42)
    
    # 四个象限
    q1 = np.random.randn(25, 2) + np.array([2, 2])
    q2 = np.random.randn(25, 2) + np.array([-2, 2])
    q3 = np.random.randn(25, 2) + np.array([-2, -2])
    q4 = np.random.randn(25, 2) + np.array([2, -2])
    
    X = np.vstack([q1, q2, q3, q4])
    # XOR: 正类在1,3象限，负类在2,4象限
    y = np.array([1]*25 + [-1]*25 + [-1]*25 + [1]*25)
    
    return X, y


def demo_nonlinear():
    print("\n" + "=" * 60)
    print("示例3：非线性可分数据 (XOR)")
    print("=" * 60)
    
    X, y = generate_xor_data()
    
    ada = AdaBoost(n_estimators=100, learning_rate=1.0)
    ada.fit(X, y)
    
    print(f"\n训练集准确率: {ada.score(X, y):.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class +1', 
                edgecolors='k', s=50)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1', 
                edgecolors='k', s=50)
    
    plt.title(f"AdaBoost XOR分类 (准确率: {ada.score(X, y):.4f})")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("adaboost_xor_demo.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demo_simple_data()
    demo_weak_learners()
    demo_nonlinear()

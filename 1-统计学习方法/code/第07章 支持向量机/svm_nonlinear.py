"""
svm_nonlinear.py

使用sklearn实现非线性SVM，用于非线性分类任务。
演示不同核函数的效果：RBF、多项式、Sigmoid

Date: 2026-03-21
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def plot_decision_boundary(X, y, model, title, subplot):
    """绘制决策边界"""
    subplot.set_title(title)
    
    # 生成网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    subplot.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50)
    
    # 标记支持向量
    if hasattr(model, 'support_vectors_'):
        subplot.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                       s=200, facecolors='none', edgecolors='green', linewidth=2,
                       label='Support Vectors')
    
    subplot.set_xlabel('特征 1')
    subplot.set_ylabel('特征 2')


# ------------------ 1. 月牙形数据（make_moons）------------------
print("=" * 60)
print("1. 月牙形数据 (make_moons) - 非线性可分")
print("=" * 60)

X_moons, y_moons = make_moons(n_samples=200, noise=0.3, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

# 不同核函数的SVM
svm_rbf_moons = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
svm_poly_moons = SVC(kernel='poly', degree=3, gamma='scale', C=1.0, random_state=42)
svm_sigmoid_moons = SVC(kernel='sigmoid', gamma='scale', C=1.0, random_state=42)

svm_rbf_moons.fit(X_train_m, y_train_m)
svm_poly_moons.fit(X_train_m, y_train_m)
svm_sigmoid_moons.fit(X_train_m, y_train_m)

print(f"RBF核 - 训练准确率: {svm_rbf_moons.score(X_train_m, y_train_m):.3f}, "
      f"测试准确率: {svm_rbf_moons.score(X_test_m, y_test_m):.3f}")
print(f"多项式核 - 训练准确率: {svm_poly_moons.score(X_train_m, y_train_m):.3f}, "
      f"测试准确率: {svm_poly_moons.score(X_test_m, y_test_m):.3f}")
print(f"Sigmoid核 - 训练准确率: {svm_sigmoid_moons.score(X_train_m, y_train_m):.3f}, "
      f"测试准确率: {svm_sigmoid_moons.score(X_test_m, y_test_m):.3f}")

# ------------------ 2. 圆形数据（make_circles）------------------
print("\n" + "=" * 60)
print("2. 圆形数据 (make_circles) - 非线性可分")
print("=" * 60)

X_circles, y_circles = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)
X_circles = StandardScaler().fit_transform(X_circles)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_circles, y_circles, test_size=0.3, random_state=42
)

svm_rbf_circles = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
svm_poly_circles = SVC(kernel='poly', degree=3, gamma='scale', C=1.0, random_state=42)

svm_rbf_circles.fit(X_train_c, y_train_c)
svm_poly_circles.fit(X_train_c, y_train_c)

print(f"RBF核 - 训练准确率: {svm_rbf_circles.score(X_train_c, y_train_c):.3f}, "
      f"测试准确率: {svm_rbf_circles.score(X_test_c, y_test_c):.3f}")
print(f"多项式核 - 训练准确率: {svm_poly_circles.score(X_train_c, y_train_c):.3f}, "
      f"测试准确率: {svm_poly_circles.score(X_test_c, y_test_c):.3f}")

# ------------------ 3. 线性可分数据（make_classification）------------------
print("\n" + "=" * 60)
print("3. 线性可分数据 (make_classification) - 线性可分")
print("=" * 60)

X_linear, y_linear = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=1,
                                         class_sep=2.0, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42
)

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf_linear = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)

svm_linear.fit(X_train_l, y_train_l)
svm_rbf_linear.fit(X_train_l, y_train_l)

print(f"线性核 - 训练准确率: {svm_linear.score(X_train_l, y_train_l):.3f}, "
      f"测试准确率: {svm_linear.score(X_test_l, y_test_l):.3f}")
print(f"RBF核 - 训练准确率: {svm_rbf_linear.score(X_train_l, y_train_l):.3f}, "
      f"测试准确率: {svm_rbf_linear.score(X_test_l, y_test_l):.3f}")

# ------------------ 可视化 ------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 月牙形数据
plot_decision_boundary(X_moons, y_moons, svm_rbf_moons, "RBF核 (月牙形)", axes[0, 0])
plot_decision_boundary(X_moons, y_moons, svm_poly_moons, "多项式核 (月牙形)", axes[0, 1])
plot_decision_boundary(X_moons, y_moons, svm_sigmoid_moons, "Sigmoid核 (月牙形)", axes[0, 2])

# 圆形数据
plot_decision_boundary(X_circles, y_circles, svm_rbf_circles, "RBF核 (圆形)", axes[1, 0])
plot_decision_boundary(X_circles, y_circles, svm_poly_circles, "多项式核 (圆形)", axes[1, 1])
# axes[1, 2] 留空

# 添加图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.suptitle("不同核函数的SVM决策边界比较", fontsize=16, y=0.98)
plt.savefig("svm_kernels_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

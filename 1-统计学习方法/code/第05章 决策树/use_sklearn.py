import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def test_pruning_classification():
    """
    测试分类树的剪枝 (预剪枝 + CCP 后剪枝)
    """
    print("=" * 80)
    print("测试决策树剪枝 (分类任务)")
    print("=" * 80)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    print(X_train)

    # --- 方法 1: 预剪枝 ---
    print("\n[方法 1] 预剪枝 (设置 max_depth, min_samples_leaf)")
    clf_pre = DecisionTreeClassifier(
        max_depth=3,  # 限制深度
        min_samples_leaf=5,  # 限制叶子最小样本
        random_state=42
    )
    clf_pre.fit(X_train, y_train)
    acc_pre = accuracy_score(y_test, clf_pre.predict(X_test))
    print(f"预剪枝准确率：{acc_pre:.2f}, 树节点数：{clf_pre.tree_.node_count}")

    # --- 方法 2: 后剪枝 (CCP 代价复杂度剪枝) ---
    print("\n[方法 2] 后剪枝 (CCP - Cost Complexity Pruning)")

    # 1. 训练完整树（不剪枝）
    clf_full = DecisionTreeClassifier(random_state=42)
    clf_full.fit(X_train, y_train)

    # 2. 获取剪枝路径
    path = clf_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    print(f"剪枝序列长度：{len(ccp_alphas)} {ccp_alphas}")  # 对应 T₀, T₁, ..., Tₙ

    # 3. 为每个 alpha 训练对应的子树
    # 注意：sklearn 内部已经实现了剪枝逻辑
    subtrees = []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        subtrees.append(clf)

    # 4. 交叉验证选择最优子树（算法 5.7 的步骤 7）
    cv_scores = []
    for clf in subtrees:
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        cv_scores.append(score)

    # 5. 选择最佳
    best_idx = np.argmax(cv_scores)
    best_alpha = ccp_alphas[best_idx]
    best_clf = subtrees[best_idx]

    print(f"最佳 alpha: {best_alpha:.5f}")
    print(f"对应子树节点数：{best_clf.tree_.node_count}")

    # 4. 用最佳 alpha 训练最终模型
    clf_post = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    clf_post.fit(X_train, y_train)
    acc_post = accuracy_score(y_test, clf_post.predict(X_test))
    print(f"后剪枝准确率：{acc_post:.2f}, 树节点数：{clf_post.tree_.node_count}")

    # --- 可视化对比 ---
    fig, axes = plt.subplots(1, 2, figsize=(30, 8))

    plot_tree(clf_pre, feature_names=iris.feature_names,
              class_names=iris.target_names, filled=True, ax=axes[0], fontsize=10)
    axes[0].set_title(f"Pre-Pruning (Depth=3)\nAcc: {acc_pre:.2f}")

    plot_tree(clf_post, feature_names=iris.feature_names,
              class_names=iris.target_names, filled=True, ax=axes[1], fontsize=10)
    axes[1].set_title(f"Post-Pruning (CCP={best_alpha:.3f})\nAcc: {acc_post:.2f}")

    plt.tight_layout()
    plt.show()


def test_pruning_regression():
    """
    测试回归树的剪枝
    """
    print("\n" + "=" * 80)
    print("测试决策树剪枝 (回归任务)")
    print("=" * 80)

    # 注意：回归任务样本量稍大一点更容易观察剪枝效果
    X, y = make_regression(n_samples=200, n_features=3, n_informative=2, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train)

    # --- 未剪枝 (容易过拟合) ---
    reg_full = DecisionTreeRegressor(random_state=42)
    reg_full.fit(X_train, y_train)
    mse_full = mean_squared_error(y_test, reg_full.predict(X_test))
    print(f"未剪枝 MSE: {mse_full:.2f}, 节点数：{reg_full.tree_.node_count}")

    # --- 预剪枝 ---
    reg_pruned = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=10,  # 回归任务中这个参数很重要，平滑预测
        random_state=42
    )
    reg_pruned.fit(X_train, y_train)
    mse_pruned = mean_squared_error(y_test, reg_pruned.predict(X_test))
    print(f"预剪枝 MSE: {mse_pruned:.2f}, 节点数：{reg_pruned.tree_.node_count}")

    # --- 可视化 ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plot_tree(reg_full, feature_names=[f'f{i}' for i in range(3)], filled=True, max_depth=3)  # 只显示前几层
    plt.title(f"Full Tree (MSE={mse_full:.1f})")

    plt.subplot(1, 2, 2)
    plot_tree(reg_pruned, feature_names=[f'f{i}' for i in range(3)], filled=True)
    plt.title(f"Pruned Tree (MSE={mse_pruned:.1f})")
    plt.show()


if __name__ == "__main__":
    test_pruning_classification()
    test_pruning_regression()
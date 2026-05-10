import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from common.dataset_load import DatasetLoader, DataProcessor

# ---------------- 配置区域 ----------------
DATASETS = {
    "sh0416___ag_news": {
        "target_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "stanfordnlp___imdb": {
        "target_names": ["Negative", "Positive"],
    },
}

# 超参数网格
N_ESTIMATORS_GRID = [50, 100, 200, 250]
MAX_DEPTH_GRID = [3, 5, 7, 10, None]
FIXED_LEARNING_RATE = 0.2          # 固定学习率

FIGURES_DIR = Path(__file__).parent / "figures"
SHOW_FIGURE = False
SAVE_FIGURE = True

# 设置 matplotlib 风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def train_once(X_train_vec, X_test_vec, y_train, y_test, n_estimators, max_depth):
    """
    训练单次 AdaBoost 模型并返回准确率
    """
    # 基分类器：决策树，深度由网格搜索决定
    base_estimator = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=FIXED_LEARNING_RATE,
        random_state=42
    )

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    return acc


def plot_adaboost_acc(ds_name, acc_matrix):
    """
    使用 matplotlib 绘制 AdaBoost 网格搜索结果
    :param ds_name: 数据集名称
    :param acc_matrix: shape (len(MAX_DEPTH_GRID), len(N_ESTIMATORS_GRID))
                       行: max_depth, 列: n_estimators
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"adaboost_depth_tuning_{ds_name}.png"

    rows, cols = acc_matrix.shape

    plt.figure()

    # 遍历每一个 max_depth (每一行)
    for i, depth in enumerate(MAX_DEPTH_GRID):
        label_text = f"Base Tree Depth: {depth}"

        # 获取当前深度下，不同 n_estimators 对应的准确率
        acc_values = acc_matrix[i, :]

        # 绘制折线图
        plt.plot(N_ESTIMATORS_GRID, acc_values, marker='o', label=label_text, linewidth=2, markersize=6)

    # 设置标题和标签
    title = f"AdaBoost Performance on {ds_name}\n(Learning Rate: {FIXED_LEARNING_RATE})"
    plt.title(title, pad=20)
    plt.xlabel("Number of Estimators (n_estimators)", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)

    # 设置 X 轴刻度
    plt.xticks(N_ESTIMATORS_GRID)

    # 添加图例
    plt.legend(title="Hyperparameters", loc="lower right")

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Figure saved to: {fig_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close()


def run_grid_search(ds_name, X_train, X_test, y_train, y_test, target_names=None):
    print(f"\n--- Starting Grid Search for {ds_name} ---")
    print(f"Fixed Learning Rate: {FIXED_LEARNING_RATE}")
    print(f"Max Depth Grid: {MAX_DEPTH_GRID}")
    print(f"N Estimators Grid: {N_ESTIMATORS_GRID}")

    # 1. 特征工程
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 2. 初始化结果矩阵
    # 行: Max Depth, 列: N Estimators
    acc_matrix = np.zeros((len(MAX_DEPTH_GRID), len(N_ESTIMATORS_GRID)))

    # 3. 执行网格搜索
    total_steps = len(MAX_DEPTH_GRID) * len(N_ESTIMATORS_GRID)
    current_step = 0

    for i, depth in enumerate(MAX_DEPTH_GRID):
        for j, n_est in enumerate(N_ESTIMATORS_GRID):
            current_step += 1
            if current_step % 5 == 0 or current_step == total_steps:
                print(f"Progress: {current_step}/{total_steps} | Depth={depth}, N_Est={n_est}")

            acc = train_once(
                X_train_vec,
                X_test_vec,
                y_train,
                y_test,
                n_estimators=n_est,
                max_depth=depth,
            )
            acc_matrix[i, j] = acc

    # 4. 打印最佳结果
    best_idx = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
    best_depth = MAX_DEPTH_GRID[best_idx[0]]
    best_n_est = N_ESTIMATORS_GRID[best_idx[1]]
    best_acc = acc_matrix[best_idx]

    print(f"\n>>> Best Result for {ds_name}:")
    print(f"    Accuracy: {best_acc:.4f}")
    print(f"    Base Tree Max Depth: {best_depth}")
    print(f"    N Estimators: {best_n_est}")

    # 使用最佳参数重新训练以生成详细的分类报告（可选）
    print("\n--- Classification Report (Best Config) ---")
    final_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=best_depth, random_state=42),
        n_estimators=best_n_est,
        learning_rate=FIXED_LEARNING_RATE,
        random_state=42
    )
    final_model.fit(X_train_vec, y_train)
    y_pred_best = final_model.predict(X_test_vec)
    if target_names:
        print(classification_report(y_test, y_pred_best, target_names=target_names))

    # 5. 绘图
    plot_adaboost_acc(ds_name, acc_matrix)


if __name__ == "__main__":
    loader = DatasetLoader()

    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        try:
            splits = loader.load(ds_name, train_ratio=0.8, eval_ratio=0.1, seed=42)
            X_train, y_train = DataProcessor.split_xy(splits["train"])
            X_test, y_test = DataProcessor.split_xy(splits["test"])

            run_grid_search(ds_name, X_train, X_test, y_train, y_test, ds_info["target_names"])

        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
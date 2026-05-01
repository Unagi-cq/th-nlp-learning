import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

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

N_ESTIMATORS_GRID = [50, 100, 200, 300, 500]
MAX_DEPTH_GRID = [3, 5, 7, 10, None]

FIGURES_DIR = Path(__file__).parent / "figures"
SHOW_FIGURE = True
SAVE_FIGURE = True

# 设置 matplotlib 风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def train_once(X_train_vec, X_test_vec, y_train, y_test, n_estimators, max_depth):
    """
    训练单次模型并返回准确率
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=32,
        random_state=42,
        verbose=0  # 减少输出噪音
    )
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return acc


def plot_grid_acc_matplotlib(ds_name, acc_matrix):
    """
    使用 matplotlib 绘制网格搜索结果
    :param ds_name: 数据集名称
    :param acc_matrix: shape (len(MAX_DEPTH_GRID), len(N_ESTIMATORS_GRID))
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"rf_tuning_{ds_name}.png"

    rows, cols = acc_matrix.shape

    # 创建图形
    plt.figure()

    # 遍历每一个 max_depth (每一行)
    for i, max_depth in enumerate(MAX_DEPTH_GRID):
        depths_label = "Unlimited" if max_depth is None else str(max_depth)
        label_text = f"Max Depth: {depths_label}"

        # 获取当前深度下，不同 n_estimators 对应的准确率
        acc_values = acc_matrix[i, :]

        # 绘制折线图
        plt.plot(N_ESTIMATORS_GRID, acc_values, marker='o', label=label_text, linewidth=2, markersize=6)

    # 设置标题和标签
    title = f"RandomForest Performance on {ds_name}"
    plt.title(title, pad=20)
    plt.xlabel("Number of Estimators (n_estimators)", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)

    # 设置 X 轴刻度为具体的 n_estimators 值
    plt.xticks(N_ESTIMATORS_GRID)

    # 添加图例
    plt.legend(title="Hyperparameters", loc="lower right")

    # 添加网格线以便阅读
    plt.grid(True, linestyle='--', alpha=0.6)

    # 自动调整布局防止标签被切掉
    plt.tight_layout()

    # 保存
    if SAVE_FIGURE:
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Figure saved to: {fig_path}")

    # 显示
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close()


def run_grid_search(ds_name, X_train, X_test, y_train, y_test):
    print(f"\n--- Starting Grid Search for {ds_name} ---")

    # 1. 特征工程 (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train matrix shape: {X_train_vec.shape}")
    print(f"Test matrix shape: {X_test_vec.shape}")

    # 2. 初始化结果矩阵
    acc_matrix = np.zeros((len(MAX_DEPTH_GRID), len(N_ESTIMATORS_GRID)))

    # 3. 执行网格搜索
    total_steps = len(MAX_DEPTH_GRID) * len(N_ESTIMATORS_GRID)
    current_step = 0

    for i, max_depth in enumerate(MAX_DEPTH_GRID):
        for j, n_estimators in enumerate(N_ESTIMATORS_GRID):
            current_step += 1
            # 简单的进度提示
            if current_step % 5 == 0 or current_step == total_steps:
                print(f"Progress: {current_step}/{total_steps}")

            acc = train_once(
                X_train_vec,
                X_test_vec,
                y_train,
                y_test,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
            acc_matrix[i, j] = acc

    # 4. 打印最佳结果概览
    best_idx = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
    best_depth = MAX_DEPTH_GRID[best_idx[0]]
    best_n_est = N_ESTIMATORS_GRID[best_idx[1]]
    best_acc = acc_matrix[best_idx]
    print(f"\n>>> Best Result for {ds_name}:")
    print(f"    Accuracy: {best_acc:.4f}")
    print(f"    Max Depth: {best_depth}")
    print(f"    N Estimators: {best_n_est}")

    # 5. 绘图
    plot_grid_acc_matplotlib(ds_name, acc_matrix)


if __name__ == "__main__":
    # 初始化加载器
    loader = DatasetLoader()

    for ds_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        try:
            # 加载数据
            splits = loader.load(ds_name, train_ratio=0.8, eval_ratio=0.1, seed=42)
            X_train, y_train = DataProcessor.split_xy(splits["train"])
            X_test, y_test = DataProcessor.split_xy(splits["test"])

            # 运行搜索
            run_grid_search(ds_name, X_train, X_test, y_train, y_test)

        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            import traceback

            traceback.print_exc()
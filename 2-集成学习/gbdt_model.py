import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from common.dataset_load import DatasetLoader, DataProcessor


DATASETS = {
    "sh0416___ag_news": {
        "target_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "stanfordnlp___imdb": {
        "target_names": ["Negative", "Positive"],
    },
}

N_ESTIMATORS_GRID = [50, 100]
MAX_DEPTH_GRID = [1, 2, 3, 4]
FIXED_LEARNING_RATE = 0.1

FIGURES_DIR = Path(__file__).parent / "figures"
SHOW_FIGURE = False
SAVE_FIGURE = True

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def train_once(X_train_vec, X_test_vec, y_train, y_test, n_estimators, max_depth):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=FIXED_LEARNING_RATE,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    return acc


def plot_gbdt_acc(ds_name, acc_matrix):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"gbdt_depth_tuning_{ds_name}.png"

    plt.figure()

    for i, depth in enumerate(MAX_DEPTH_GRID):
        label_text = f"Tree Depth: {depth}"
        acc_values = acc_matrix[i, :]
        plt.plot(N_ESTIMATORS_GRID, acc_values, marker='o', label=label_text, linewidth=2, markersize=6)

    title = f"GBDT Performance on {ds_name}\n(Learning Rate: {FIXED_LEARNING_RATE})"
    plt.title(title, pad=20)
    plt.xlabel("Number of Estimators (n_estimators)", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.xticks(N_ESTIMATORS_GRID)
    plt.legend(title="Hyperparameters", loc="lower right")
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

    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    acc_matrix = np.zeros((len(MAX_DEPTH_GRID), len(N_ESTIMATORS_GRID)))

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

    best_idx = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
    best_depth = MAX_DEPTH_GRID[best_idx[0]]
    best_n_est = N_ESTIMATORS_GRID[best_idx[1]]
    best_acc = acc_matrix[best_idx]

    print(f"\n>>> Best Result for {ds_name}:")
    print(f"    Accuracy: {best_acc:.4f}")
    print(f"    Tree Max Depth: {best_depth}")
    print(f"    N Estimators: {best_n_est}")

    print("\n--- Classification Report (Best Config) ---")
    final_model = GradientBoostingClassifier(
        n_estimators=best_n_est,
        learning_rate=FIXED_LEARNING_RATE,
        max_depth=best_depth,
        random_state=42,
    )
    final_model.fit(X_train_vec, y_train)
    y_pred_best = final_model.predict(X_test_vec)
    if target_names:
        print(classification_report(y_test, y_pred_best, target_names=target_names))

    plot_gbdt_acc(ds_name, acc_matrix)


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

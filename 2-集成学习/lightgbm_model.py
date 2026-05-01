import lightgbm as lgb
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


def train(X_train, X_test, y_train, y_test, target_names=None):
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    num_class = len(set(y_train))
    train_data = lgb.Dataset(X_train_vec, label=y_train)

    if num_class == 2:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": 7,
            "learning_rate": 0.1,
            "verbose": -1,
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        y_pred_prob = model.predict(X_test_vec)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        params = {
            "objective": "multiclass",
            "num_class": num_class,
            "metric": "multi_logloss",
            "max_depth": 7,
            "learning_rate": 0.1,
            "verbose": -1,
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        y_pred_prob = model.predict(X_test_vec)
        y_pred = y_pred_prob.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"LightGBM Accuracy: {acc:.4f}")
    if target_names:
        print(classification_report(y_test, y_pred, target_names=target_names))
    return acc


if __name__ == "__main__":
    loader = DatasetLoader()

    for ds_name, ds_info in DATASETS.items():
        print(f"\n>>> Dataset: {ds_name}")
        print("=" * 60)

        splits = loader.load(ds_name, train_ratio=0.8, eval_ratio=0.1, seed=42)
        X_train, y_train = DataProcessor.split_xy(splits["train"])
        X_test, y_test = DataProcessor.split_xy(splits["test"])

        train(X_train, X_test, y_train, y_test, ds_info["target_names"])

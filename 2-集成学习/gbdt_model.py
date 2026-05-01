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


def train(X_train, X_test, y_train, y_test, target_names=None):
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train_vec.toarray(), y_train)

    y_pred = model.predict(X_test_vec.toarray())
    acc = accuracy_score(y_test, y_pred)
    print(f"GBDT Accuracy: {acc:.4f}")
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

import csv
import json
from pathlib import Path
from random import Random


class DatasetLoader:
    """只负责加载数据集并按比例返回 train/eval/test 元组数据。"""

    def __init__(self, base_dir=None):
        """
        初始化加载器。

        Args:
            base_dir (str | Path | None): 数据根目录的上级目录。None 表示当前文件所在目录。
        """
        root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.dataset_dir = root / "datasets"

    @staticmethod
    def _read_json(path):
        """
        读取 JSON 文件；若不是标准 JSON，则按 JSONL 回退解析。

        Args:
            path (str | Path): 文件路径。

        Returns:
            list[dict]: 记录列表。
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return data if isinstance(data, list) else [data]

    @staticmethod
    def _read_jsonl(path):
        """
        读取 JSONL 文件。

        Args:
            path (str | Path): 文件路径。

        Returns:
            list[dict]: 记录列表。
        """
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _read_csv(path):
        """
        读取 text,label 两列的 CSV 文件。

        Args:
            path (str | Path): 文件路径。

        Returns:
            tuple[list[str], list[int]]: 文本与标签。
        """
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["text"])
                labels.append(int(row["label"]))
        return texts, labels

    def _load_raw(self, dataset_name):
        """
        按数据集文件夹名加载原始文本和标签。

        Args:
            dataset_name (str): 数据集文件夹名。

        Returns:
            tuple[list[tuple[str, int]], list[tuple[str, int]] | None]:
                第一个为训练池样本，第二个为原生测试集样本（若无则为 None）。
        """
        if dataset_name == "sh0416___ag_news":
            train_records = self._read_jsonl(self.dataset_dir / dataset_name / "train.jsonl")
            test_records = self._read_jsonl(self.dataset_dir / dataset_name / "test.jsonl")
            train_pairs = [(f"{x['title']} {x['description']}", int(x["label"]) - 1) for x in train_records]
            test_pairs = [(f"{x['title']} {x['description']}", int(x["label"]) - 1) for x in test_records]
            return train_pairs, test_pairs

        if dataset_name == "stanfordnlp___imdb":
            train_texts, train_labels = self._read_csv(self.dataset_dir / dataset_name / "train.json")
            test_records = self._read_json(self.dataset_dir / dataset_name / "test.json")
            test_texts = [x["text"] for x in test_records]
            test_labels = [int(x["label"]) for x in test_records]
            train_pairs = list(zip(train_texts, train_labels))
            test_pairs = list(zip(test_texts, test_labels))
            return train_pairs, test_pairs

        raise ValueError(f"Unsupported dataset: {dataset_name}")

    @staticmethod
    def _split_pairs(pairs, train_ratio, eval_ratio, seed):
        """
        对 (text, label) 元组列表做打乱和切分。

        Args:
            pairs (list[tuple[str, int]]): 全量样本。
            train_ratio (float): 训练集比例。
            eval_ratio (float): 验证集比例。
            seed (int): 随机种子。

        Returns:
            dict[str, list[tuple[str, int]]]: train/eval/test 三个切分。
        """
        if not (0 < train_ratio < 1 and 0 <= eval_ratio < 1 and train_ratio + eval_ratio < 1):
            raise ValueError("Invalid split ratios.")

        idx = list(range(len(pairs)))
        Random(seed).shuffle(idx)
        shuffled = [pairs[i] for i in idx]

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)

        return {
            "train": shuffled[:n_train],
            "eval": shuffled[n_train:n_train + n_eval],
            "test": shuffled[n_train + n_eval:],
        }

    @staticmethod
    def _split_train_eval(train_pairs, train_ratio, eval_ratio, seed):
        """
        在已有 test 的前提下，仅对 train 池切分 train/eval。

        Args:
            train_pairs (list[tuple[str, int]]): 原生训练样本。
            train_ratio (float): 用户输入的训练比例（与 eval 一起归一化）。
            eval_ratio (float): 用户输入的验证比例（与 train 一起归一化）。
            seed (int): 随机种子。

        Returns:
            tuple[list[tuple[str, int]], list[tuple[str, int]]]: 切分后的 train 与 eval。
        """
        if not (train_ratio > 0 and eval_ratio >= 0 and train_ratio + eval_ratio > 0):
            raise ValueError("train_ratio and eval_ratio must be valid when dataset already has test split.")

        idx = list(range(len(train_pairs)))
        Random(seed).shuffle(idx)
        shuffled = [train_pairs[i] for i in idx]

        train_share = train_ratio / (train_ratio + eval_ratio)
        n_train = int(len(shuffled) * train_share)
        return shuffled[:n_train], shuffled[n_train:]

    def load(self, dataset_name, train_ratio=0.8, eval_ratio=0.1, seed=42):
        """
        加载数据并返回元组样本切分。

        Args:
            dataset_name (str): 数据集文件夹名。
            train_ratio (float): 训练集比例。
            eval_ratio (float): 验证集比例。
            seed (int): 随机种子。

        Returns:
            dict[str, list[tuple[str, int]]]: train/eval/test，每条样本为 (text, label)。
        """
        train_pool, test_pool = self._load_raw(dataset_name)
        if test_pool is not None:
            train_split, eval_split = self._split_train_eval(train_pool, train_ratio, eval_ratio, seed)
            return {"train": train_split, "eval": eval_split, "test": test_pool}
        return self._split_pairs(train_pool, train_ratio, eval_ratio, seed)


class DataProcessor:
    """负责对原始元组数据做特征转化。"""

    @staticmethod
    def split_xy(samples):
        """
        将 (text, label) 元组样本拆分为 X 和 y。

        Args:
            samples (list[tuple[str, int]]): 元组样本列表。

        Returns:
            tuple[list[str], list[int]]: 文本列表和标签列表。
        """
        texts = [x for x, _ in samples]
        labels = [y for _, y in samples]
        return texts, labels

    @staticmethod
    def tfidf(train_samples, eval_samples=None, test_samples=None, max_features=2000):
        """
        使用 TF-IDF 对样本做特征化。

        Args:
            train_samples (list[tuple[str, int]]): 训练样本。
            eval_samples (list[tuple[str, int]] | None): 验证样本。
            test_samples (list[tuple[str, int]] | None): 测试样本。
            max_features (int): 最大特征数。

        Returns:
            dict[str, tuple]: 每个切分返回 (X_vec, y)。
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        x_train, y_train = DataProcessor.split_xy(train_samples)
        vectorizer = TfidfVectorizer(max_features=max_features)
        out = {"train": (vectorizer.fit_transform(x_train), y_train)}

        if eval_samples is not None:
            x_eval, y_eval = DataProcessor.split_xy(eval_samples)
            out["eval"] = (vectorizer.transform(x_eval), y_eval)
        if test_samples is not None:
            x_test, y_test = DataProcessor.split_xy(test_samples)
            out["test"] = (vectorizer.transform(x_test), y_test)
        return out

    @staticmethod
    def llm_tokenize(samples, tokenizer_name="bert-base-uncased", max_length=256):
        """
        使用 LLM tokenizer 将文本转为 token ids。

        Args:
            samples (list[tuple[str, int]]): 元组样本列表。
            tokenizer_name (str): HuggingFace tokenizer 名称。
            max_length (int): 最大截断长度。

        Returns:
            tuple[list[list[int]], list[int]]: token ids 和标签。
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError("Please install transformers to use llm_tokenize.") from e

        texts, labels = DataProcessor.split_xy(samples)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        encoded = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        return encoded["input_ids"], labels


if __name__ == "__main__":
    loader = DatasetLoader()

    ag = loader.load("sh0416___ag_news", train_ratio=0.8, eval_ratio=0.1, seed=42)
    print("ag_news:", len(ag["train"]), len(ag["eval"]), len(ag["test"]))

    imdb = loader.load("stanfordnlp___imdb", train_ratio=0.8, eval_ratio=0.1, seed=42)
    print("imdb:", len(imdb["train"]), len(imdb["eval"]), len(imdb["test"]))

    x_train, y_train = DataProcessor.split_xy(ag["train"])
    print("sample:", y_train[0], x_train[0][:80])

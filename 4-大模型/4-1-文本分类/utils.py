import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common.dataset_load import DatasetLoader


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_data(dataset):
    data = DatasetLoader().load(dataset)
    n_classes = max(y for _, y in data["train"]) + 1
    sizes = {split: len(samples) for split, samples in data.items()}
    print(f"dataset {dataset}  classes {n_classes}  sizes {sizes}")

    lengths = [len(t.split()) for t, _ in data["train"]]
    print(f"train len (words)  min {min(lengths)}  max {max(lengths)}  mean {sum(lengths) / len(lengths):.1f}  median {median(lengths):.1f}")
    return data, n_classes

from pathlib import Path

from .common import classifier_build

MODEL_DIR = Path(__file__).resolve().parents[2] / "pre_training_models" / "bert-base-uncased"

DEFAULT_CFG = dict(
    max_len=32,
    epochs=3,
    batch=32,
    lr=5e-6,
    freeze=False,  # True 时冻结 BERT 主干，只训练分类头
    eval_steps=200,
    patience=5,
)


build = classifier_build(MODEL_DIR)

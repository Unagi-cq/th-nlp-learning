from pathlib import Path

from .common import classifier_build

MODEL_DIR = Path(__file__).resolve().parents[2] / "pre_training_models" / "gpt2"

DEFAULT_CFG = dict(
    max_len=32,    # GPT2 无双向注意力，标题类短文本可适当放宽
    epochs=3,
    batch=32,
    lr=5e-6,
    freeze=False,  # True 时冻结 GPT2 主干，只训练分类头
    eval_steps=200,
    patience=5,
)


def configure(tokenizer, model):
    """GPT-2 没有 pad token，使用 eos 补齐批次。"""
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


build = classifier_build(MODEL_DIR, configure=configure)

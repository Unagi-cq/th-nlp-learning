"""分类模型的默认实现；模型专属设置通过 configure(tokenizer, model) 定制。

如需自定义编码或预测方式，可在模型文件中直接实现 build(cfg, device, n_classes)，
返回 tokenizer、model、enc、predict，与训练入口保持相同接口。
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def make_encoder(tokenizer, device, max_len):
    """统一签名 enc(texts, labels=None) -> batch dict。"""

    def enc(texts, labels=None):
        tokens = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        batch = {k: v.to(device) for k, v in tokens.items()}
        if labels is not None:
            batch["labels"] = torch.tensor(labels).to(device)
        return batch

    return enc


def make_predictor(model, enc):
    """取分类 logits 最大的类别。"""

    def predict(texts):
        model.eval()
        with torch.no_grad():
            return model(**enc(texts)).logits.argmax(1).tolist()

    return predict


def classifier_build(model_dir, configure=None):
    """创建默认 build；configure 在加载后、冻结主干前执行，可原地修改分词器和模型。"""

    def build(cfg, device, n_classes):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=n_classes, attn_implementation="eager"
        )
        if configure is not None:
            configure(tokenizer, model)
        if cfg.get("freeze", False):
            model.base_model.requires_grad_(False)
        model.to(device)
        enc = make_encoder(tokenizer, device, cfg["max_len"])
        predict = make_predictor(model, enc)
        return tokenizer, model, enc, predict

    return build

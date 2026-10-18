import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import get_device, load_data

MODEL_DIR = Path(__file__).resolve().parents[1] / "pre_training_models" / "bert-base-uncased"


def build_model(model_dir, n_classes, device, freeze_bert=False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=n_classes, attn_implementation="eager"  # MPS 上 SDPA 不支持 dropout
    )
    if freeze_bert:
        model.bert.requires_grad_(False)
    model.to(device)
    return tokenizer, model


def make_encoder(tokenizer, device, max_len):
    def enc(texts):
        t = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        return {k: v.to(device) for k, v in t.items()}

    return enc


def make_predictor(model, enc):
    def predict(texts):
        model.eval()
        with torch.no_grad():
            return model(**enc(texts)).logits.argmax(1).tolist()

    return predict


def evaluate(predict, samples, batch_size=64):
    texts, labels = zip(*samples)
    preds = []
    for i in range(0, len(texts), batch_size):
        preds.extend(predict(list(texts[i:i + batch_size])))
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train(model, enc, predict, data, device, save_path, epochs, batch, lr, eval_steps, patience):
    x_train = [t for t, _ in data["train"]]
    y_train = [l for _, l in data["train"]]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr)

    best_f1, patience_left, step = -1.0, patience, 0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), batch):
            idx = perm[i:i + batch]
            batch_data = enc([x_train[j] for j in idx])
            yb = torch.tensor([y_train[j] for j in idx]).to(device)
            opt.zero_grad()
            loss = model(**batch_data, labels=yb).loss
            loss.backward()
            opt.step()
            step += 1
            print(f"epoch {epoch + 1}/{epochs}  step {step}  samples {i + len(idx)}/{len(x_train)}  loss {loss.item():.4f}")
            if step % eval_steps == 0:
                val = evaluate(predict, data["eval"], 128)
                improved = val["macro_f1"] > best_f1
                print(f"step {step}  train_loss {loss.item():.4f}  val_acc {val['accuracy']:.4f}  val_f1 {val['macro_f1']:.4f}{' *' if improved else ''}")
                if improved:
                    best_f1, patience_left = val["macro_f1"], patience
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    print(f"saved checkpoint -> {save_path}")
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        print(f"early stop: {patience} 次评估无提升，best val_f1 {best_f1:.4f}")
                        break
                model.train()
        if patience_left == 0:
            break


if __name__ == "__main__":
    cfg = dict(
        dataset="sh0416___ag_news",  # 或 "stanfordnlp___imdb"
        max_len=32,
        epochs=3,
        batch=128,
        lr=2e-5,
        freeze_bert=True,  # True 时冻结 BERT 主干，只训练分类头
        eval_steps=200,  # 每多少 step 评估一次
        patience=5,     # 连续多少次评估无提升就早停
    )

    data, n_classes = load_data(cfg["dataset"])
    device = get_device()

    tokenizer, model = build_model(MODEL_DIR, n_classes, device, cfg["freeze_bert"])
    enc = make_encoder(tokenizer, device, cfg["max_len"])

    predict = make_predictor(model, enc)

    save_path = Path(__file__).resolve().parent / "checkpoints" / f"{cfg['dataset']}_best.pt"
    train(model, enc, predict, data, device, save_path, cfg["epochs"], cfg["batch"], cfg["lr"], cfg["eval_steps"],
          cfg["patience"])

    model.load_state_dict(torch.load(save_path))
    print(evaluate(predict, data["test"]))

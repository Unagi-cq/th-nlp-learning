import torch
from sklearn.metrics import accuracy_score, f1_score

from utils import get_device, load_data


def evaluate(model, enc, predict, samples, batch_size=64):
    """评估：loss 用 model(**enc(texts, labels)).loss，preds 用 predict。"""
    texts, labels = zip(*samples)
    total_loss, preds = 0.0, []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i + batch_size])
            batch_labels = list(labels[i:i + batch_size])
            total_loss += model(**enc(batch_texts, batch_labels)).loss.item() * len(batch_texts)
            preds.extend(predict(batch_texts))
    return {
        "loss": total_loss / len(texts),
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train(model, enc, predict, data, device, save_path, epochs, batch, lr, eval_steps, patience, log_dir=None):
    """通用训练循环。enc(texts, labels) 已把 labels 放进 batch，故 loss 统一从 model(**batch).loss 取。"""
    from collections import deque
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir) if log_dir else None
    recent_losses = deque(maxlen=50)  # 记录最近 50 步 loss，算滑动平均
    x_train = [t for t, _ in data["train"]]
    y_train = [l for _, l in data["train"]]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr)

    best_f1, patience_left, step = -1.0, patience, 0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), batch):
            idx = perm[i:i + batch]
            batch_data = enc([x_train[j] for j in idx], [y_train[j] for j in idx])
            opt.zero_grad()
            loss = model(**batch_data).loss
            loss.backward()
            opt.step()
            step += 1
            current_loss = loss.item()
            recent_losses.append(current_loss)
            print(f"epoch {epoch + 1}/{epochs}  step {step}  samples {i + len(idx)}/{len(x_train)}  loss {current_loss:.4f}")
            if writer:
                writer.add_scalar("train/loss", current_loss, step)
                writer.add_scalar("train/loss_smooth", sum(recent_losses) / len(recent_losses), step)
            if step % eval_steps == 0:
                val = evaluate(model, enc, predict, data["eval"], 128)
                improved = val["macro_f1"] > best_f1
                print(f"step {step}  train_loss {current_loss:.4f}  val_loss {val['loss']:.4f}  val_acc {val['accuracy']:.4f}  val_f1 {val['macro_f1']:.4f}{' *' if improved else ''}")
                if writer:
                    writer.add_scalar("val/loss", val["loss"], step)
                    writer.add_scalar("val/accuracy", val["accuracy"], step)
                    writer.add_scalar("val/macro_f1", val["macro_f1"], step)
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

    if writer:
        writer.close()


def run(cfg, model_name, build):
    """通用主流程：加载数据 -> 构建模型/编码器/预测器 -> 训练 -> 测试集评估 -> 写 results.csv。

    Args:
        cfg (dict): 超参配置，至少含 dataset/epochs/batch/lr/eval_steps/patience/max_len。
        model_name (str): checkpoint 文件名前缀。
        build (callable): build(cfg, device, n_classes) -> (tokenizer, model, enc, predict)。
    """
    from datetime import datetime
    from pathlib import Path

    data, n_classes = load_data(cfg["dataset"])
    device = get_device()
    tokenizer, model, enc, predict = build(cfg, device, n_classes)

    print(model)

    base = Path(__file__).resolve().parent
    save_path = base / "checkpoints" / f"{model_name}_{cfg['dataset']}_best.pt"
    log_dir = base / "runs" / f"{model_name}_{cfg['dataset']}_{datetime.now():%m%d_%H%M%S}"
    train(model, enc, predict, data, device, save_path,
          cfg["epochs"], cfg["batch"], cfg["lr"], cfg["eval_steps"], cfg["patience"], str(log_dir))

    model.load_state_dict(torch.load(save_path))
    test_metrics = evaluate(model, enc, predict, data["test"])
    print(test_metrics)

    save_results(cfg, model_name, log_dir.name, test_metrics)


def save_results(cfg, model_name, runs_dir, metrics):
    """把实验参数、runs 目录名、测试集效果追加写入 results.csv。"""
    import csv
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent / "results.csv"
    fields = ["model", "dataset", "lr", "batch", "max_len", "epochs", "freeze",
              "runs_dir", "test_loss", "test_acc", "test_f1"]
    row = {
        "model": model_name,
        "dataset": cfg["dataset"],
        "lr": cfg["lr"],
        "batch": cfg["batch"],
        "max_len": cfg["max_len"],
        "epochs": cfg["epochs"],
        "freeze": cfg.get("freeze", False),
        "runs_dir": runs_dir,
        "test_loss": f"{metrics['loss']:.4f}",
        "test_acc": f"{metrics['accuracy']:.4f}",
        "test_f1": f"{metrics['macro_f1']:.4f}",
    }
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"已写入实验记录 -> {csv_path}")


if __name__ == "__main__":
    """
    用法：
        1. 选模型：改 model（"bert" / "gpt2" / "distilbert"）。
        2. 选数据集：改 dataset（"sh0416___ag_news" / "stanfordnlp___imdb"）。
        3. 训练：  python train.py
           模型脚本在 models/ 下，各自定义 DEFAULT_CFG 和 build，特殊设置通过 configure 定制。
        4. 可视化：训练会往 runs/{model}_{dataset}_{时间戳}/ 写 TensorBoard 日志，
           另开终端运行 `cd 4-大模型/4-1-文本分类 && tensorboard --logdir runs`，浏览器打开 http://localhost:6006 实时看曲线。

    记录指标：
        train/loss         每个 step 的训练 loss（瞬时值，锯齿明显）
        train/loss_smooth  最近 50 步 loss 的滑动平均，便于看趋势
        val/loss           每个 eval_steps 的验证集 loss（判断过拟合）
        val/accuracy       每个 eval_steps 的验证集准确率
        val/macro_f1       每个 eval_steps 的验证集 macro-F1
    """
    from models import bert, distilbert, gpt2

    MODELS = {
        "bert": (bert.DEFAULT_CFG, bert.build),
        "gpt2": (gpt2.DEFAULT_CFG, gpt2.build),
        "distilbert": (distilbert.DEFAULT_CFG, distilbert.build),
    }

    model = "bert"
    dataset = "sh0416___ag_news"  # 或 "stanfordnlp___imdb"

    default_cfg, build = MODELS[model]
    cfg = {**default_cfg, "dataset": dataset}
    run(cfg, model, build)

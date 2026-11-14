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
    """通用训练循环；返回验证集 macro-F1 最佳 checkpoint 对应的 step。"""
    from collections import deque
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir) if log_dir else None
    recent_losses = deque(maxlen=50)  # 记录最近 50 步 loss，算滑动平均
    x_train = [t for t, _ in data["train"]]
    y_train = [l for _, l in data["train"]]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr)

    best_f1, patience_left, step = -1.0, patience, 0
    best_step = 0
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
            is_last_step = epoch == epochs - 1 and i + batch >= len(x_train)
            if step % eval_steps == 0 or is_last_step:
                val = evaluate(model, enc, predict, data["eval"], 128)
                improved = val["macro_f1"] > best_f1
                print(f"step {step}  train_loss {current_loss:.4f}  val_loss {val['loss']:.4f}  val_acc {val['accuracy']:.4f}  val_f1 {val['macro_f1']:.4f}{' *' if improved else ''}")
                if writer:
                    writer.add_scalar("val/loss", val["loss"], step)
                    writer.add_scalar("val/accuracy", val["accuracy"], step)
                    writer.add_scalar("val/macro_f1", val["macro_f1"], step)
                if improved:
                    best_f1, patience_left = val["macro_f1"], patience
                    best_step = step
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
    if best_step == 0:
        raise RuntimeError("未产生有效的最佳 checkpoint，请检查训练数据、epochs 和验证结果。")
    print(f"best val_f1 {best_f1:.4f}  best_step {best_step}")
    return best_step


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
    best_step = train(model, enc, predict, data, device, save_path,
                      cfg["epochs"], cfg["batch"], cfg["lr"], cfg["eval_steps"], cfg["patience"], str(log_dir))

    model.load_state_dict(torch.load(save_path))
    test_metrics = evaluate(model, enc, predict, data["test"])
    print(test_metrics)

    save_results(cfg, model_name, log_dir.name, test_metrics, best_step)


def save_results(cfg, model_name, runs_dir, metrics, best_step):
    """记录全部 cfg、最佳 step 和测试指标；新增配置自动扩列，保留历史记录。"""
    import csv
    import json
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    csv_path = Path(__file__).resolve().parent / "results.csv"
    row = {
        "model": model_name,
        "runs_dir": runs_dir,
        "best_step": best_step,
        "test_loss": f"{metrics['loss']:.4f}",
        "test_acc": f"{metrics['accuracy']:.4f}",
        "test_f1": f"{metrics['macro_f1']:.4f}",
    }
    # 配置与结果字段重名时加 cfg_ 前缀，避免丢失任一值。
    used_fields = set(row) | set(cfg)
    for key, value in cfg.items():
        column = key
        if column in row:
            column = f"cfg_{column}"
            while column in used_fields:
                column = f"cfg_{column}"
        used_fields.add(column)
        row[column] = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list, tuple)) or value is None else value

    old_fields, old_rows = [], []
    if csv_path.exists() and csv_path.stat().st_size:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields, old_rows = reader.fieldnames or [], list(reader)
    fields = list(dict.fromkeys([*old_fields, *row]))
    temp_path = None
    try:
        with NamedTemporaryFile("w", newline="", encoding="utf-8", dir=csv_path.parent, delete=False) as f:
            temp_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row)
        temp_path.replace(csv_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    print(f"已写入实验记录 -> {csv_path}")


if __name__ == "__main__":
    """
    用法：
        1. 选模型：改 model（"bert" / "gpt2" / "distilbert" / "roberta"）。
        2. 选数据集：改 dataset（"sh0416___ag_news" / "stanfordnlp___imdb"）。
           max_len 随数据集自动设置：AG News 为 32，IMDb 为 256，所有模型共用。
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
    from models import bert, distilbert, gpt2, roberta

    MODELS = {
        "bert": (bert.DEFAULT_CFG, bert.build),
        "gpt2": (gpt2.DEFAULT_CFG, gpt2.build),
        "distilbert": (distilbert.DEFAULT_CFG, distilbert.build),
        "roberta": (roberta.DEFAULT_CFG, roberta.build),
    }
    MAX_LEN_BY_DATASET = {
        "sh0416___ag_news": 32,
        "stanfordnlp___imdb": 256,
    }

    model = "bert"
    dataset = "sh0416___ag_news"  # 或 "stanfordnlp___imdb"

    default_cfg, build = MODELS[model]
    cfg = {**default_cfg, "dataset": dataset, "max_len": MAX_LEN_BY_DATASET[dataset]}
    run(cfg, model, build)

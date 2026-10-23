from pathlib import Path

from modelscope import snapshot_download

OUT_DIR = Path(__file__).resolve().parent

# 换成 t5-small / distilgpt2 会更快、更省显存，适合先在 MPS 上试跑
MODELS = {
    "bert-base-uncased": "AI-ModelScope/bert-base-uncased",
    "distilbert-base-uncased": "AI-ModelScope/distilbert-base-uncased",
    "t5-base": "AI-ModelScope/t5-base",
    "gpt2": "AI-ModelScope/gpt2",
}

# 只保留 PyTorch 的 safetensors 权重，跳过 flax/tf/rust/bin 等其他框架的大文件
IGNORE = [r"(flax_model\.msgpack|tf_model\.h5|rust_model\.ot|pytorch_model\.bin)$"]

for name, model_id in MODELS.items():
    print(f"下载 {name} <- {model_id}")
    snapshot_download(model_id, local_dir=str(OUT_DIR / name), ignore_file_pattern=IGNORE)
    print(f"完成 -> {OUT_DIR / name}")

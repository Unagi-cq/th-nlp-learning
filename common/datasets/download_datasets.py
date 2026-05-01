from datasets import load_dataset

# 下载并加载数据集 (例如: glue, imdb, common_voice 等)
# cache_dir 可以指定缓存路径，避免默认下载到 ~/.cache
dataset = load_dataset("stanfordnlp/imdb", cache_dir="./")

# 查看数据结构
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: ...
#     test: ...
# })

# 访问具体数据
train_data = dataset['train']
print(train_data[0])

# 导出为 CSV
dataset['train'].to_csv("train.json")

# 导出为 JSON
dataset['test'].to_json("test.json")
dataset['unsupervised'].to_json("unsupervised.json")
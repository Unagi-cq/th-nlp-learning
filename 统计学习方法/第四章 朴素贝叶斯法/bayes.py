"""
bayes.py

实现朴素贝叶斯算法，用于文本分类任务。

说明：
- 使用多项式朴素贝叶斯模型
- 采用拉普拉斯平滑（Laplace smoothing）避免零概率问题
- 假设特征（词）之间相互独立

Author: Huang CQ
Date: 2026-01-26
"""
import numpy as np
from collections import Counter, defaultdict


class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        初始化朴素贝叶斯分类器
        :param alpha: 拉普拉斯平滑参数，默认为1.0
        """
        self.alpha = alpha  # 拉普拉斯平滑参数
        self.vocab = None  # 词表
        self.vocab_size = 0  # 词表大小
        self.class_prior = {}  # 先验概率 P(Y)
        self.class_word_count = defaultdict(lambda: defaultdict(int))  # 每个类别下每个词的计数
        self.class_total_words = defaultdict(int)  # 每个类别下的总词数
        self.classes = None  # 所有类别

    def _build_vocab(self, sentences):
        """
        构建词表
        :param sentences: 句子列表，每个句子是词列表
        :return: 词表（字典，词->索引）
        """
        all_words = []
        for sentence in sentences:
            all_words.extend(sentence)
        vocab = {word: idx for idx, word in enumerate(set(all_words))}
        print(f"构建词表，共 {len(vocab)} 个词")
        print(f"词表: {list(vocab.keys())}")
        return vocab

    def fit(self, X, y):
        """
        训练朴素贝叶斯模型
        :param X: 训练数据，list of list，每个元素是一个句子（词列表）
        :param y: 标签列表，每个元素是一个类别
        """
        print("=" * 60)
        print("开始训练朴素贝叶斯模型")
        print("=" * 60)
        
        # 构建词表
        self.vocab = self._build_vocab(X)
        self.vocab_size = len(self.vocab)
        self.classes = list(set(y))
        
        print(f"\n类别: {self.classes}")
        print(f"训练样本数: {len(X)}")
        print(f"词表大小: {self.vocab_size}\n")
        
        # 统计每个类别的样本数
        class_count = Counter(y)
        total_samples = len(X)
        
        # 计算先验概率 P(Y)
        print("计算先验概率 P(Y):")
        for cls in self.classes:
            self.class_prior[cls] = class_count[cls] / total_samples
            print(f"  P({cls}) = {class_count[cls]}/{total_samples} = {self.class_prior[cls]:.4f}")
        
        # 统计每个类别下每个词的出现次数
        print("\n统计每个类别下每个词的出现次数:")
        for i, (sentence, label) in enumerate(zip(X, y)):
            print(f"\n样本 {i+1}: {sentence}, 类别: {label}")
            for word in sentence:
                if word in self.vocab:
                    self.class_word_count[label][word] += 1
                    self.class_total_words[label] += 1
                    print(f"  词 '{word}' 在类别 {label} 中出现，当前计数: {self.class_word_count[label][word]}")
        
        # 打印统计结果
        print("\n" + "=" * 60)
        print("统计结果汇总:")
        print("=" * 60)
        for cls in self.classes:
            print(f"\n类别 {cls}:")
            print(f"  总词数: {self.class_total_words[cls]}")
            print(f"  词频统计:")
            for word, count in sorted(self.class_word_count[cls].items()):
                print(f"    '{word}': {count} 次")
        
        print("\n训练完成！")

    def _calculate_word_probability(self, word, label):
        """
        计算 P(word|label)，使用拉普拉斯平滑
        :param word: 词
        :param label: 类别
        :return: 条件概率
        """
        # 拉普拉斯平滑: P(word|label) = (count(word, label) + alpha) / (total_words(label) + alpha * vocab_size)
        word_count = self.class_word_count[label].get(word, 0)
        total_words = self.class_total_words[label]
        prob = (word_count + self.alpha) / (total_words + self.alpha * self.vocab_size)
        return prob

    def predict(self, X):
        """
        预测类别
        :param X: 测试数据，list of list，每个元素是一个句子（词列表）
        :return: 预测的类别列表
        """
        predictions = []
        
        print("\n" + "=" * 60)
        print("开始预测")
        print("=" * 60)
        
        for i, sentence in enumerate(X):
            print(f"\n预测样本 {i+1}: {sentence}")
            
            # 对每个类别计算后验概率 P(label|sentence) ∝ P(label) * ∏P(word|label)
            class_scores = {}
            
            for label in self.classes:
                # 先验概率
                log_prob = np.log(self.class_prior[label])
                print(f"\n  类别 {label}:")
                print(f"    log P({label}) = {log_prob:.4f}")
                
                # 条件概率（使用对数避免下溢）
                word_probs = []
                for word in sentence:
                    if word in self.vocab:
                        word_prob = self._calculate_word_probability(word, label)
                        log_word_prob = np.log(word_prob)
                        word_probs.append(log_word_prob)
                        print(f"    log P('{word}'|{label}) = {log_word_prob:.4f} (P = {word_prob:.6f})")
                    else:
                        # 词不在词表中，使用平滑概率
                        word_prob = self.alpha / (self.class_total_words[label] + self.alpha * self.vocab_size)
                        log_word_prob = np.log(word_prob)
                        word_probs.append(log_word_prob)
                        print(f"    log P('{word}'|{label}) = {log_word_prob:.4f} (词不在词表中，使用平滑)")
                
                # 累加对数概率
                log_prob += sum(word_probs)
                class_scores[label] = log_prob
                print(f"    总对数概率: {log_prob:.4f}")
            
            # 选择概率最大的类别
            predicted_label = max(class_scores, key=class_scores.get)
            predictions.append(predicted_label)
            print(f"\n  → 预测类别: {predicted_label} (对数概率: {class_scores[predicted_label]:.4f})")
        
        return predictions


# ------------------ 示例：文本分类 ------------------
if __name__ == "__main__":
    # 构造10个句子的数据集（每个句子是词列表）
    # 这里做一个简单的情感分类示例：正面(positive)和负面(negative)
    sentences = [
        ["我", "喜欢", "这个", "电影"],
        ["这个", "电影", "很", "好看"],
        ["我", "爱", "看", "电影"],
        ["这个", "电影", "太", "棒了"],
        ["我", "觉得", "这个", "电影", "不错"],
        ["我", "讨厌", "这个", "电影"],
        ["这个", "电影", "很", "无聊"],
        ["我", "不", "喜欢", "看", "电影"],
        ["这个", "电影", "太", "差了"],
        ["我", "觉得", "这个", "电影", "很", "糟糕"]
    ]
    
    # 对应的标签：前5个是正面，后5个是负面
    labels = ["positive", "positive", "positive", "positive", "positive",
              "negative", "negative", "negative", "negative", "negative"]
    
    print("数据集:")
    for i, (sent, label) in enumerate(zip(sentences, labels)):
        print(f"  {i+1}. {sent} -> {label}")
    
    # 训练朴素贝叶斯模型
    nb = NaiveBayes(alpha=1.0)
    nb.fit(sentences, labels)
    
    # 测试预测
    test_sentences = [
        ["我", "喜欢", "电影"],  # 应该预测为 positive
        ["这个", "电影", "很", "差"],  # 应该预测为 negative
        ["电影", "很", "好看"],  # 应该预测为 positive
    ]
    
    print("\n" + "=" * 60)
    print("测试预测")
    print("=" * 60)
    predictions = nb.predict(test_sentences)
    
    print("\n" + "=" * 60)
    print("预测结果汇总:")
    print("=" * 60)
    for sent, pred in zip(test_sentences, predictions):
        print(f"  {sent} -> {pred}")


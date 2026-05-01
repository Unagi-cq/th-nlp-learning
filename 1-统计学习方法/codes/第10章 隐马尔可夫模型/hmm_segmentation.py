"""
hmm_segmentation.py

使用隐马尔可夫模型进行中文分词

Date: 2026-03-22
"""

import numpy as np
from typing import List, Tuple, Dict
import re


class HMMSegmenter:
    """
    基于HMM的中文分词器
    
    状态: 
        B - 词语开头
        M - 词语中间
        E - 词语结尾
        S - 单字词语
    
    观测: 汉字
    """
    
    def __init__(self):
        self.states = ['B', 'M', 'E', 'S']
        self.n_states = len(self.states)
        self.state2Idx = {s: i for i, s in enumerate(self.states)}
        
        self.pi = np.array([0.5, 0.1, 0.1, 0.3])  # 初始概率
        self.A = np.array([                        # 状态转移概率
            [0.5, 0.3, 0.2, 0.0],  # B -> B, M, E, S
            [0.0, 0.4, 0.6, 0.0],  # M -> B, M, E, S
            [0.6, 0.0, 0.0, 0.4],  # E -> B, M, E, S
            [0.5, 0.0, 0.0, 0.5],  # S -> B, M, E, S
        ])
        
        self.B = None  # 观测概率，训练后设置
        self.vocab_size = 0
        
    def train(self, sentences: List[List[str]], words: List[List[str]]):
        """
        使用训练数据估计模型参数
        
        :param sentences: 分字后的句子列表
        :param words: 分词后的词列表
        """
        # 统计观测概率
        obs_count = {s: {} for s in self.states}
        total_count = {s: 0 for s in self.states}
        
        for sent_chars, sent_words in zip(sentences, words):
            char_idx = 0
            for word in sent_words:
                if len(word) == 1:
                    state = 'S'
                else:
                    state = 'B'
                    obs_count[state][word[0]] = obs_count[state].get(word[0], 0) + 1
                    total_count[state] += 1
                    
                    for i in range(1, len(word) - 1):
                        state = 'M'
                        obs_count[state][word[i]] = obs_count[state].get(word[i], 0) + 1
                        total_count[state] += 1
                        
                    state = 'E'
                    obs_count[state][word[-1]] = obs_count[state].get(word[-1], 0) + 1
                    total_count[state] += 1
                char_idx += len(word)
        
        # 构建词汇表
        vocab = set()
        for sent in sentences:
            vocab.update(sent)
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        self.char2Idx = {c: i for i, c in enumerate(self.vocab)}
        
        # 构建观测概率矩阵
        self.B = np.ones((self.n_states, self.vocab_size)) * 1e-10  # 拉普拉斯平滑
        
        for state in self.states:
            for char, count in obs_count[state].items():
                if char in self.char2Idx:
                    self.B[self.state2Idx[state], self.char2Idx[char]] = count / total_count[state]
        
        print(f"训练完成! 词汇量: {self.vocab_size}")
    
    def get_state_sequence(self, word: str) -> List[str]:
        """获取词语对应的状态序列"""
        if len(word) == 1:
            return ['S']
        else:
            return ['B'] + ['M'] * (len(word) - 2) + ['E']
    
    def sentence_to_observation(self, sentence: str) -> List[int]:
        """将句子转换为观测序列"""
        chars = list(sentence)
        obs = []
        for c in chars:
            if c in self.char2Idx:
                obs.append(self.char2Idx[c])
            else:
                obs.append(0)  # 未知字符
        return obs
    
    def viterbi(self, observations: np.ndarray) -> List[str]:
        """
        维特比算法分词
        
        :param observations: 观测序列
        :return: 状态序列
        """
        T = len(observations)
        
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化
        obs_idx = observations[0]
        for i in range(self.n_states):
            delta[0, i] = self.pi[i] * self.B[i, obs_idx]
        
        # 递推
        for t in range(1, T):
            obs_idx = observations[t]
            for j in range(self.n_states):
                max_prob = 0
                max_state = 0
                for i in range(self.n_states):
                    prob = delta[t-1, i] * self.A[i, j]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                delta[t, j] = max_prob * self.B[j, obs_idx]
                psi[t, j] = max_state
        
        # 回溯
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return [self.states[p] for p in path]
    
    def segment(self, sentence: str) -> List[str]:
        """
        对句子进行分词
        
        :param sentence: 输入句子
        :return: 分词结果列表
        """
        observations = self.sentence_to_observation(sentence)
        states = self.viterbi(observations)
        
        words = []
        current_word = ""
        
        for i, (char, state) in enumerate(zip(sentence, states)):
            if state == 'B':
                if current_word:
                    words.append(current_word)
                current_word = char
            elif state == 'M':
                current_word += char
            elif state == 'E':
                current_word += char
                words.append(current_word)
                current_word = ""
            elif state == 'S':
                if current_word:
                    words.append(current_word)
                words.append(char)
                current_word = ""
        
        if current_word:
            words.append(current_word)
        
        return words


def load_data():
    """加载示例训练数据"""
    # 简化的训练数据：分字版本 -> 分词版本
    训练数据 = [
        ("我 爱 中 国", ["我", "爱", "中国"]),
        ("我 喜欢 学习 机器 学习", ["我", "喜欢", "学习", "机器", "学习"]),
        ("他 在 研究 自然 语言 处理", ["他", "在", "研究", "自然", "语言", "处理"]),
        ("北京 是 首都", ["北京", "是", "首都"]),
        ("上海 是 一个 大 城市", ["上海", "是", "一个", "大", "城市"]),
        ("我 爱 北京 天安门", ["我", "爱", "北京", "天安门"]),
        ("自然 语言 处理 很 有趣", ["自然", "语言", "处理", "很", "有趣"]),
        ("机器 学习 是 人工智能 的 核心", ["机器", "学习", "是", "人工智能", "的", "核心"]),
        ("深 度 学习 取得 重大 进展", ["深度", "学习", "取得", "重大", "进展"]),
        ("我 喜欢 美妙 的 音乐", ["我", "喜欢", "美妙", "的", "音乐"]),
    ]
    
    sentences = []
    words = []
    
    for sentence, seg in 训练数据:
        # 清理空格
        sentence = sentence.replace(" ", "")
        sentences.append(list(sentence))
        words.append(seg)
    
    return sentences, words


def evaluate(segmenter: HMMSegmenter, test_sentences: List[str], correct_segments: List[List[str]]):
    """评估分词效果"""
    print("\n" + "=" * 60)
    print("分词结果评估")
    print("=" * 60)
    
    correct = 0
    total = 0
    
    for sent, correct_seg in zip(test_sentences, correct_segments):
        pred_seg = segmenter.segment(sent)
        
        print(f"\n句子: {sent}")
        print(f"预测: {' / '.join(pred_seg)}")
        print(f"正确: {' / '.join(correct_seg)}")
        
        # 简单评估：词数是否匹配
        if pred_seg == correct_seg:
            correct += 1
            print("  ✓ 完全正确!")
        else:
            # 检查是否有相同词
            for p, c in zip(pred_seg, correct_seg):
                if p == c:
                    correct += 1
                total += 1
    
    print(f"\n准确率: {correct}/{total} = {correct/total*100:.2f}%")


def main():
    """主函数"""
    print("=" * 60)
    print("基于HMM的中文分词示例")
    print("=" * 60)
    
    # 加载训练数据
    sentences, words = load_data()
    
    print(f"\n训练数据:")
    for i, (sent, seg) in enumerate(zip(sentences, words)):
        print(f"  {i+1}. {''.join(sent)} -> {' / '.join(seg)}")
    
    # 训练模型
    segmenter = HMMSegmenter()
    segmenter.train(sentences, words)
    
    # 测试分词
    test_cases = [
        "我喜欢上了中国",
        "我喜欢学习机器学习",
        "北京是中国的首都",
        "自然语言处理很有意思",
        "深度学习是人工智能的核心",
    ]
    
    print("\n" + "=" * 60)
    print("分词测试")
    print("=" * 60)
    
    for sentence in test_cases:
        result = segmenter.segment(sentence)
        print(f"\n句子: {sentence}")
        print(f"分词: {' / '.join(result)}")
    
    # 更详细的例子
    print("\n" + "-" * 60)
    print("详细示例：北京是首都")
    print("-" * 60)
    
    sentence = "北京是首都"
    observations = segmenter.sentence_to_observation(sentence)
    states = segmenter.viterbi(np.array(observations))
    
    print(f"句子: {sentence}")
    print(f"字符: {list(sentence)}")
    print(f"状态: {states}")
    print(f"分词: {' / '.join(segmenter.segment(sentence))}")


if __name__ == "__main__":
    main()

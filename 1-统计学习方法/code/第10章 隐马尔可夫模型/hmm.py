"""
hmm.py

实现隐马尔可夫模型（Hidden Markov Model），包括：
- 概率计算：前向算法、后向算法
- 参数学习：最大似然估计
- 解码：维特比算法

Date: 2026-03-22
"""

import numpy as np
from typing import List, Tuple


class HiddenMarkovModel:
    """
    隐马尔可夫模型
    
    状态空间: Q = {1, 2, ..., N}
    观测空间: V = {1, 2, ..., M}
    初始状态概率: π
    状态转移概率: A
    观测概率: B
    """
    
    def __init__(self, n_states: int, n_observations: int):
        """
        初始化HMM模型
        
        :param n_states: 状态数量
        :param n_observations: 观测数量
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # 模型参数
        self.pi = None  # 初始状态概率分布
        self.A = None   # 状态转移概率矩阵
        self.B = None   # 观测概率矩阵
        
    def initialize_parameters(self, 
                              pi: np.ndarray = None,
                              A: np.ndarray = None,
                              B: np.ndarray = None):
        """
        初始化模型参数
        
        :param pi: 初始状态概率向量 (n_states,)
        :param A: 状态转移概率矩阵 (n_states, n_states)
        :param B: 观测概率矩阵 (n_states, n_observations)
        """
        if pi is not None:
            self.pi = pi
        else:
            self.pi = np.ones(self.n_states) / self.n_states
            
        if A is not None:
            self.A = A
        else:
            self.A = np.ones((self.n_states, self.n_states)) / self.n_states
            
        if B is not None:
            self.B = B
        else:
            self.B = np.ones((self.n_states, self.n_observations)) / self.n_observations
    
    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        前向算法计算观测序列概率
        
        :param observations: 观测序列 (T,)
        :return: alpha矩阵 (T, n_states), P(O|λ)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # 初始化
        alpha[0] = self.pi * self.B[:, observations[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * self.B[j, observations[t]]
        
        # 终止
        prob = np.sum(alpha[T-1])
        
        return alpha, prob
    
    def backward(self, observations: np.ndarray) -> np.ndarray:
        """
        后向算法计算观测序列概率
        
        :param observations: 观测序列 (T,)
        :return: beta矩阵 (T, n_states)
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # 初始化
        beta[T-1] = 1
        
        # 递推
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1])
        
        return beta
    
    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        维特比算法求最优状态序列
        
        :param observations: 观测序列 (T,)
        :return: 最优状态序列 (T,), 概率
        """
        T = len(observations)
        
        # 初始化
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = self.pi * self.B[:, observations[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1] * self.A[:, j]) * self.B[j, observations[t]]
                psi[t, j] = np.argmax(delta[t-1] * self.A[:, j])
        
        # 终止
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        prob = delta[T-1, path[T-1]]
        
        # 回溯
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, prob
    
    def generate(self, length: int) -> np.ndarray:
        """
        生成观测序列
        
        :param length: 序列长度
        :return: 观测序列 (length,)
        """
        observations = np.zeros(length, dtype=int)
        
        # 生成初始状态
        state = np.random.choice(self.n_states, p=self.pi)
        observations[0] = np.random.choice(self.n_observations, p=self.B[state])
        
        # 生成后续状态和观测
        for t in range(1, length):
            state = np.random.choice(self.n_states, p=self.A[state])
            observations[t] = np.random.choice(self.n_observations, p=self.B[state])
        
        return observations


def demo_weather_prediction():
    """
    天气预测示例
    
    状态: [晴天, 多云, 雨天]
    观测: [散步, 购物, 清洗]
    """
    print("=" * 60)
    print("示例1: 天气预测")
    print("=" * 60)
    
    # 状态空间: 0=晴天, 1=多云, 2=雨天
    n_states = 3
    # 观测空间: 0=散步, 1=购物, 2=清洗
    n_observations = 3
    
    hmm = HiddenMarkovModel(n_states, n_observations)
    
    # 初始化参数
    # 初始状态概率：晴天最常见
    pi = np.array([0.5, 0.3, 0.2])
    
    # 状态转移概率
    A = np.array([
        [0.5, 0.3, 0.2],  # 晴天->晴天, 晴天->多云, 晴天->雨天
        [0.3, 0.4, 0.3],  # 多云->晴天, 多云->多云, 多云->雨天
        [0.2, 0.3, 0.5]   # 雨天->晴天, 雨天->多云, 雨天->雨天
    ])
    
    # 观测概率
    B = np.array([
        [0.6, 0.3, 0.1],  # 晴天时: 散步0.6, 购物0.3, 清洗0.1
        [0.4, 0.4, 0.2],  # 多云时: 散步0.4, 购物0.4, 清洗0.2
        [0.1, 0.2, 0.7]   # 雨天时: 散步0.1, 购物0.2, 清洗0.7
    ])
    
    hmm.initialize_parameters(pi, A, B)
    
    print("\n模型参数:")
    print(f"状态数量: {n_states}, 观测数量: {n_observations}")
    print(f"\n初始状态概率 π: {pi}")
    print(f"\n状态转移概率矩阵 A:")
    print(A)
    print(f"\n观测概率矩阵 B:")
    print(B)
    
    # 预测天气
    # 观测序列: 散步, 购物, 清洗, 散步
    observations = np.array([0, 1, 2, 0])
    obs_names = ['散步', '购物', '清洗']
    
    print(f"\n观测序列: {', '.join([obs_names[o] for o in observations])}")
    
    # 前向算法
    alpha, prob = hmm.forward(observations)
    print(f"\n前向算法计算的概率 P(O|λ) = {prob:.6f}")
    
    # 后向算法验证
    beta = hmm.backward(observations)
    # 用后向算法计算概率
    prob_backward = np.sum(hmm.pi * hmm.B[:, observations[0]] * beta[0])
    print(f"后向算法计算的概率 P(O|λ) = {prob_backward:.6f}")
    
    # 维特比算法找最优状态序列
    path, path_prob = hmm.viterbi(observations)
    state_names = ['晴天', '多云', '雨天']
    print(f"\n最优状态序列: {', '.join([state_names[s] for s in path])}")
    print(f"该序列的概率: {path_prob:.6f}")
    
    # 生成示例序列
    print("\n" + "-" * 60)
    print("生成示例天气序列:")
    for i in range(3):
        gen_obs = hmm.generate(5)
        gen_path, _ = hmm.viterbi(gen_obs)
        print(f"  示例 {i+1}:")
        print(f"    观测: {', '.join([obs_names[o] for o in gen_obs])}")
        print(f"    状态: {', '.join([state_names[s] for s in gen_path])}")


def demo_pony_tones():
    """
    小马国女孩发音示例
    
    状态: [E, G, K, P, S, T]
    观测: different tone patterns
    """
    print("\n" + "=" * 60)
    print("示例2: 小马国女孩发音")
    print("=" * 60)
    
    n_states = 6
    n_observations = 4
    
    hmm = HiddenMarkovModel(n_states, n_observations)
    
    # 简化的参数
    pi = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    
    A = np.array([
        [0.3, 0.2, 0.1, 0.1, 0.2, 0.1],
        [0.1, 0.3, 0.2, 0.1, 0.2, 0.1],
        [0.1, 0.1, 0.3, 0.2, 0.1, 0.2],
        [0.2, 0.1, 0.1, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.2, 0.1, 0.3, 0.1],
        [0.2, 0.1, 0.1, 0.2, 0.1, 0.3]
    ])
    
    B = np.array([
        [0.4, 0.3, 0.2, 0.1],  # E
        [0.3, 0.3, 0.2, 0.2],  # G
        [0.2, 0.2, 0.3, 0.3],  # K
        [0.3, 0.2, 0.2, 0.3],  # P
        [0.2, 0.3, 0.3, 0.2],  # S
        [0.1, 0.2, 0.4, 0.3]   # T
    ])
    
    hmm.initialize_parameters(pi, A, B)
    
    print("\n模型参数:")
    print(f"状态数量: {n_states}, 观测数量: {n_observations}")
    print(f"\n初始状态概率 π: {pi}")
    
    # 观测序列
    observations = np.array([0, 1, 2, 3, 0])
    obs_names = [' Tone 1', ' Tone 2', ' Tone 3', ' Tone 4']
    state_names = ['E', 'G', 'K', 'P', 'S', 'T']
    
    print(f"\n观测序列: {', '.join([obs_names[o] for o in observations])}")
    
    # 维特比解码
    path, path_prob = hmm.viterbi(observations)
    print(f"\n最优状态序列: {'-'.join([state_names[s] for s in path])}")
    print(f"该序列的概率: {path_prob:.6f}")


def demo_comparison():
    """
    比较不同观测序列的概率
    """
    print("\n" + "=" * 60)
    print("示例3: 不同序列概率比较")
    print("=" * 60)
    
    # 使用天气预测的模型
    n_states = 3
    n_observations = 3
    
    hmm = HiddenMarkovModel(n_states, n_observations)
    
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array([
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    B = np.array([
        [0.6, 0.3, 0.1],
        [0.4, 0.4, 0.2],
        [0.1, 0.2, 0.7]
    ])
    
    hmm.initialize_parameters(pi, A, B)
    
    state_names = ['晴天', '多云', '雨天']
    obs_names = ['散步', '购物', '清洗']
    
    # 测试不同序列
    test_sequences = [
        [0, 0, 0],  # 连续散步
        [2, 2, 2],  # 连续清洗
        [0, 1, 2],  # 散步->购物->清洗
    ]
    
    print("\n观测序列概率比较:")
    print("-" * 60)
    for obs in test_sequences:
        _, prob = hmm.forward(obs)
        path, path_prob = hmm.viterbi(obs)
        print(f"\n观测: {', '.join([obs_names[o] for o in obs])}")
        print(f"  前向概率: {prob:.6f}")
        print(f"  最优路径: {', '.join([state_names[s] for s in path])}")
        print(f"  路径概率: {path_prob:.6f}")


if __name__ == "__main__":
    # 运行所有示例
    demo_weather_prediction()
    demo_pony_tones()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)

# -*- coding:gbk -*-

"""
经验缓冲区
"""

import numpy as np
import torch
from typing import Union
import gymnasium
from ._sum_tree import SumTree

__all__ = ["TrajectoryBuffer",
           "SimpleReplayBuffer",
           "RolloutBuffer",
           "PrioritizedReplayBuffer"]


class BaseBuffer(object):
    """
    基本缓冲区
    """
    def __init__(self, device: str):
        """
        构造函数
        """
        self.device = torch.device(device)

    def store(self, *args, **kwargs):
        """
        存储数据
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def clear(self):
        """
        清除缓冲区数据
        :return:
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """
        从缓冲区采样数据并转换为张量
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def recall(self, *args, **kwargs):
        """
        输出缓冲区所有数据
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        对于多环境，交换轴0 (n_steps）和轴1（num_envs)，
        并且将形状[n_steps, n_envs, ...]转换为[n_steps * n_envs, ...]
        优先按照时间顺序排列
        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        将数组转换为张量
        :param arr:
        :return:
        """
        return torch.tensor(arr, device=self.device, dtype=torch.float32)


class RolloutBufferSampler(object):
    """
    数据采样器
    """
    def __init__(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 log_probs: torch.Tensor,
                 values,
                 advantages,
                 returns):
        """

        :param states:
        :param actions:
        :param log_probs:
        :param values:
        :param advantages:
        :param returns:
        """
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.values = values
        self.advantages = advantages
        self.returns = returns



class TrajectoryBuffer(BaseBuffer):
    """
    n_step自举轨迹缓冲区
    """

    def __init__(self,
                 state_dim: tuple | int,
                 action_num: int,
                 capacity: int = 2000,
                 gamma: float = 0.99,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        构造函数
        :param state_dim:
        :param action_num:
        :param capacity:
        :param gamma:
        :param device:
        """
        super(TrajectoryBuffer, self).__init__(device)

        # 原始奖励
        self.rewards = np.zeros((capacity, ), dtype=np.float32)
        # 动作
        self.actions = np.zeros((capacity, action_num), dtype=np.float32)
        # 状态值
        if isinstance(state_dim, int):
            state_dim = (state_dim, )
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        # 奖励折扣系数
        self.gamma = gamma

        # 指针位置
        self.idx = 0
        # 是否填满
        self.full = False
        # 最大长度
        self.capacity = capacity

    def clear(self):
        """
        清除轨迹数据
        :return:
        """
        # 重置为0
        self.rewards.fill(0)
        self.actions.fill(0)
        self.states.fill(0)
        # 重置指针位置
        self.idx = 0
        # 重置状态
        self.full = False

    def store(self,
              state: np.ndarray,
              action: Union[np.ndarray, int],
              reward: float,
              ):
        """
        存储经验元组
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 奖励
        :return:
        """
        if not self.full:
            # 存储
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward

            # 指针挪到下一位
            self.idx += 1
            # 判断是否满了
            if self.idx >= self.capacity:
                self.full = True

    def recall(self, v_next: torch.Tensor = None):
        """
        提取轨迹数据
        :param v_next: 下一时刻的状态值
        :return:
        """
        # 状态轨迹
        state = self.to_tensor(self.states[:self.idx])
        # 动作轨迹
        action = self.to_tensor(self.actions[:self.idx])
        # 折扣奖励
        discounted_reward = self.to_tensor(self.compute_returns(v_next))

        return state, action, discounted_reward

    def compute_returns(self, v_next: torch.Tensor = None):
        """
        计算折扣回报
        :param v_next:下一时刻的状态值
        :return:
        """
        # 转换为numpy
        last_reward = v_next.squeeze().detach().cpu().numpy() if v_next is not None \
            else np.array(0, dtype=np.float32)
        # 折扣回报
        discounted_reward = np.zeros((self.idx, ), dtype=np.float32)

        for step in reversed(range(self.idx)):
            # 计算折扣奖励
            last_reward = self.rewards[step] + self.gamma * last_reward
            # 存储
            discounted_reward[step] = last_reward

        return discounted_reward

    def __len__(self):
        """
        获取轨迹长度
        :return:
        """
        return self.idx


class SimpleReplayBuffer(BaseBuffer):
    """
    经验回放缓冲区
    """

    def __init__(self,
                 state_dim: tuple | int,
                 action_num: int,
                 capacity: int,
                 batch_size: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        构造函数
        :param state_dim: 状态维度
        :param action_num: 动作个数
        :param capacity: 缓冲区大小
        :param batch_size: 批大小
        :param device: 张量设备
        """
        super(SimpleReplayBuffer, self).__init__(device)

        assert capacity > batch_size, f"缓冲区容量要大于采样的批大小"

        # 基本结构
        self.capacity = capacity
        self.batch_size = batch_size

        # 内存
        if isinstance(state_dim, int):
            state_dim = (state_dim, )
        self.states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_num), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, ), dtype=np.float32)
        self.dones = np.zeros((self.capacity, ), dtype=np.float32)

        # 当前存放的位置
        self.idx = 0
        # 已经使用长度
        self.length = 0

    def store(self,
              state: np.ndarray,
              action: Union[np.ndarray, int],
              next_state: np.ndarray,
              reward: float,
              done: bool) -> None:
        """
        存储经验
        :param state: 当前状态
        :param action: 动作
        :param next_state: 下一个状态
        :param reward: 奖励
        :param done: 是否结束
        :return:
        """
        # 存储
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        # 指针位置加1
        self.idx += 1
        self.idx = self.idx % self.capacity

        # 已经使用长度加1
        self.length += 1
        if self.length >= self.capacity:
            self.length = self.capacity

    def sample(self) -> tuple:
        """
        采样
        :return:
        """
        # 采样的索引
        index = np.random.choice(self.length, size=self.batch_size, replace=False)

        # 变为张量
        state = self.to_tensor(self.states[index])
        action = self.to_tensor(self.actions[index])
        next_state = self.to_tensor(self.next_states[index])
        reward = self.to_tensor(self.rewards[index]).unsqueeze(-1)
        done = self.to_tensor(self.dones[index]).unsqueeze(-1)

        return state, action, next_state, reward, done

    def __len__(self) -> int:
        """
        获取当前缓冲区已使用长度
        :return:
        """
        return self.length


class RolloutBuffer(BaseBuffer):
    """
    并行环境轨迹生成抽样缓冲区
    """

    def __init__(self,
                 state_dim: tuple | int,
                 action_num: int,
                 env: gymnasium.vector.VectorEnv,
                 capacity: int,
                 batch_size: int = None,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        构造函数
        :param state_dim: 状态维度
        :param action_num: 动作个数
        :param env: 并行环境
        :param capacity: 缓冲区容量
        :param batch_size: 采样批大小
        :param gamma: 奖励折扣系数
        :param gae_lambda: GAE折扣系数
        :param device: 张量设备
        """
        super(RolloutBuffer, self).__init__(device)

        # 基本参数
        self.state_dim = state_dim if isinstance(state_dim, tuple) else (state_dim, )
        self.action_num = action_num
        self.num_envs = env.num_envs
        self.env = env
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 是否满了
        self.full = False
        # 当前位置
        self.idx = 0
        # 记录轨迹模拟结束的时候的状态和结束标识
        self.last_state = None
        self.last_done = None

        # 内存
        self.states = np.zeros((self.capacity, self.num_envs, *self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.num_envs, self.action_num), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_envs), dtype=np.float32)

    def clear(self):
        """
        清除数据
        :return:
        """
        # 批量数据归零
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self.dones.fill(0)

        # 重置标识符
        self.full = False
        self.idx = 0

    def store(self,
              state: np.ndarray,
              action: np.ndarray,
              reward: np.ndarray,
              log_prob: np.ndarray,
              value: np.ndarray,
              done: np.ndarray):
        """
        存储数据
        :param state:
        :param action:
        :param reward:
        :param log_prob:
        :param value:
        :param done:
        :return:
        """
        if not self.full:
            # 修正维度
            if len(action.shape) == 1:
                action = np.expand_dims(action, -1)
            if len(log_prob.shape) == 2:
                log_prob = np.squeeze(log_prob, -1)
            if len(value.shape) == 2:
                value = np.squeeze(value, -1)
            # 存储
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward
            self.log_probs[self.idx] = log_prob
            self.values[self.idx] = value
            self.dones[self.idx] = done
            # 指针后移
            self.idx += 1

            # 是否存储满
            if self.idx >= self.capacity:
                # 存储满了标志位
                self.full = True

    def compute_returns_and_advantage(self, v_next: torch.Tensor):
        """
        计算GAE和折扣回报
        :return:
        """
        # 下一时刻的V值
        v_next = v_next.squeeze().detach().cpu().numpy()
        # 下一时刻的单步自举值
        last_gae_lam = np.zeros((self.num_envs, ), dtype=np.float32)
        # 下一时刻的奖励
        last_reward = v_next

        for step in reversed(range(self.capacity)):
            # 交互结束标识
            non_terminal = 1.0 - self.dones[step]
            # 下一时刻状态值
            if step == self.capacity - 1:
                next_values = v_next
            else:
                next_values = self.values[step + 1]
            # 计算单步自举
            delta = self.rewards[step] + self.gamma * next_values * non_terminal - self.values[step]
            # 计算GAE
            last_gae_lam = delta + self.gamma * self.gae_lambda * non_terminal * last_gae_lam
            # 计算折扣奖励
            last_reward = self.rewards[step] + self.gamma * non_terminal * last_reward
            # 存储
            self.advantages[step] = last_gae_lam
            # 计算折扣回报
            self.returns[step] = last_reward

    def roll_out(self,
                 agent,
                 seed: int = None,
                 ):
        """
        轨迹模拟
        :param agent: 智能体
        :param seed: 环境种子
        :return:
        """
        # 从上一次模拟结束的位置开始
        state = self.last_state
        done = self.last_done

        # 启动步，重置环境
        if state is None and done is None:
            state, _ = self.env.reset()
            done = np.array([False for _ in range(self.num_envs)], dtype=bool)

        while not self.full:
            # 选择动作
            action, log_pa, _ = agent.select_action(state, "train")
            # 计算状态值
            value = agent.value_model(self.to_tensor(state))
            # 交互
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # 结束标识
            done = np.logical_or(terminated, truncated)
            # 存储
            self.store(state, action, reward, log_pa.detach().cpu().numpy(),
                       value.detach().cpu().numpy(), done.astype(np.float32))
            # 更新状态
            state = next_state.copy()

            # 重置环境
            if np.any(done):
                state, _ = self.env.reset(seed=seed, options={"reset_mask": done})

        # 计算状态值
        next_value = agent.value_model(self.to_tensor(state))

        # 计算折扣回报和GAE
        self.compute_returns_and_advantage(next_value)

        # 记录模拟结束位置
        self.last_state = state.copy()
        self.last_done = done.copy()

    def sample(self, total: bool = False):
        """
        轨迹采样
        :param total: 是否输出所有轨迹
        :return:
        """
        # 采样长度
        batch_size = self.batch_size if not total else self.capacity * self.num_envs

        # 展平，并且转换为张量
        state = self.to_tensor(self.swap_and_flatten(self.states))
        action = self.to_tensor(self.swap_and_flatten(self.actions))
        log_prob = self.to_tensor(self.swap_and_flatten(self.log_probs))
        value = self.to_tensor(self.swap_and_flatten(self.values))
        advantage = self.to_tensor(self.swap_and_flatten(self.advantages))
        returns = self.to_tensor(self.swap_and_flatten(self.returns))

        # 起始索引
        start_idx = 0

        # 按照索引选择数据
        while start_idx < self.capacity * self.num_envs:
            trajectory = (state[start_idx: start_idx + batch_size],
                          action[start_idx: start_idx + batch_size],
                          log_prob[start_idx: start_idx + batch_size],
                          value[start_idx: start_idx + batch_size],
                          advantage[start_idx: start_idx + batch_size],
                          returns[start_idx: start_idx + batch_size])
            yield trajectory
            start_idx += batch_size


class PrioritizedReplayBuffer(BaseBuffer):
    """
    优先经验回放缓冲区
    """
    def __init__(self,
                 capacity: int,
                 batch_size: int,
                 alpha: float,
                 beta: float,
                 beta_inc_step: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        构造函数
        :param capacity:
        :param batch_size:
        :param alpha:
        :param beta:
        :param beta_inc_step:
        :param device:
        """
        super(PrioritizedReplayBuffer, self).__init__(device)

        assert 0 <= alpha <= 1, f"alpha 要在[0, 1]区间内，实际为{alpha}"
        assert 0 <= beta < 1, f"beta 要在[0, 1)区间内，实际为{beta}"

        # 存储传入参数
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_inc_step = beta_inc_step
        # beta列表
        self.beta_list = np.linspace(beta, 1.0, beta_inc_step)
        # 当前长度
        self.length = 0

        # 内存
        self.memory = SumTree(capacity)

    def _get_priority(self, td_error):
        """
        计算优先级
        :param td_error:
        :return:
        """
        return (np.abs(td_error) + 1e-6) ** self.alpha

    def store(self,
              td_error: float,
              state: np.ndarray,
              action: int,
              next_state: np.ndarray,
              reward: float,
              done: bool
              ) -> None:
        """
        存储经验
        :param td_error: TD误差
        :param state: 当前状态
        :param action: 动作
        :param next_state: 下一个状态
        :param reward: 奖励
        :param done: 是否结束
        :return:
        """
        # 计算优先级
        p = self._get_priority(td_error)
        # 按照优先级存储
        self.memory.add(p, (state, action, next_state, reward, done))

    def sample(self, size: int = None):
        """
        采样
        :param size: 采样批大小
        :return:
        """
        batched_data = []
        index = []
        priorities = []

        # 采样大小
        bs = size if size is not None else self.batch_size

        segment = self.memory.total_priority / bs

        for i in range(bs):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            # 获取叶子节点
            idx, p, data = self.memory.get_leaf(v)
            batched_data.append(data)
            index.append(idx)
            priorities.append(p)

        # 计算采样概率
        sampling_probabilities = np.array(priorities) / self.memory.total_priority
        # 计算权重
        weights = (len(self.memory.data) * sampling_probabilities) ** (-self.beta)
        # 权重归一化
        weights /= weights.max()
        # 转变为tensor
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # 聚合为Tensor类型
        state = torch.tensor(np.array([batched_data[i][0] for i in range(bs)]),
                             dtype=torch.float32, device=self.device)
        action = torch.tensor(np.array([batched_data[i][1] for i in range(bs)]),
                              dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array([batched_data[i][2] for i in range(bs)]),
                                  dtype=torch.float32, device=self.device)
        reward = torch.tensor(np.array([batched_data[i][3] for i in range(bs)]), dtype=torch.float32,
                              device=self.device).unsqueeze(-1)
        done = torch.tensor(np.array([batched_data[i][4] for i in range(bs)]), dtype=torch.float32,
                            device=self.device).unsqueeze(-1)

        # 更新步长
        self.length += 1
        # 更新beta
        if self.length > len(self.beta_list):
            self.beta = 1.0
        else:
            self.beta = self.beta_list[self.length]

        return index, weights, state, action, next_state, reward, done

    def update(self, idx, td_errors):
        for idx, td_error in zip(idx, td_errors):
            # 计算优先级
            p = self._get_priority(td_error)
            # 更新优先级
            self.memory.update(idx, p)

    def __len__(self):
        """
        获取缓冲区长度
        :return:
        """
        return self.length

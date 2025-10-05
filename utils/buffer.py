# -*- coding:gbk -*-

"""
���黺����
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
    ����������
    """
    def __init__(self, device: str):
        """
        ���캯��
        """
        self.device = torch.device(device)

    def store(self, *args, **kwargs):
        """
        �洢����
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def clear(self):
        """
        �������������
        :return:
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """
        �ӻ������������ݲ�ת��Ϊ����
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def recall(self, *args, **kwargs):
        """
        �����������������
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        ���ڶ໷����������0 (n_steps������1��num_envs)��
        ���ҽ���״[n_steps, n_envs, ...]ת��Ϊ[n_steps * n_envs, ...]
        ���Ȱ���ʱ��˳������
        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        ������ת��Ϊ����
        :param arr:
        :return:
        """
        return torch.tensor(arr, device=self.device, dtype=torch.float32)


class RolloutBufferSampler(object):
    """
    ���ݲ�����
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
    n_step�Ծٹ켣������
    """

    def __init__(self,
                 state_dim: tuple | int,
                 action_num: int,
                 capacity: int = 2000,
                 gamma: float = 0.99,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        ���캯��
        :param state_dim:
        :param action_num:
        :param capacity:
        :param gamma:
        :param device:
        """
        super(TrajectoryBuffer, self).__init__(device)

        # ԭʼ����
        self.rewards = np.zeros((capacity, ), dtype=np.float32)
        # ����
        self.actions = np.zeros((capacity, action_num), dtype=np.float32)
        # ״ֵ̬
        if isinstance(state_dim, int):
            state_dim = (state_dim, )
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        # �����ۿ�ϵ��
        self.gamma = gamma

        # ָ��λ��
        self.idx = 0
        # �Ƿ�����
        self.full = False
        # ��󳤶�
        self.capacity = capacity

    def clear(self):
        """
        ����켣����
        :return:
        """
        # ����Ϊ0
        self.rewards.fill(0)
        self.actions.fill(0)
        self.states.fill(0)
        # ����ָ��λ��
        self.idx = 0
        # ����״̬
        self.full = False

    def store(self,
              state: np.ndarray,
              action: Union[np.ndarray, int],
              reward: float,
              ):
        """
        �洢����Ԫ��
        :param state: ��ǰ״̬
        :param action: ��ǰ����
        :param reward: ����
        :return:
        """
        if not self.full:
            # �洢
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward

            # ָ��Ų����һλ
            self.idx += 1
            # �ж��Ƿ�����
            if self.idx >= self.capacity:
                self.full = True

    def recall(self, v_next: torch.Tensor = None):
        """
        ��ȡ�켣����
        :param v_next: ��һʱ�̵�״ֵ̬
        :return:
        """
        # ״̬�켣
        state = self.to_tensor(self.states[:self.idx])
        # �����켣
        action = self.to_tensor(self.actions[:self.idx])
        # �ۿ۽���
        discounted_reward = self.to_tensor(self.compute_returns(v_next))

        return state, action, discounted_reward

    def compute_returns(self, v_next: torch.Tensor = None):
        """
        �����ۿۻر�
        :param v_next:��һʱ�̵�״ֵ̬
        :return:
        """
        # ת��Ϊnumpy
        last_reward = v_next.squeeze().detach().cpu().numpy() if v_next is not None \
            else np.array(0, dtype=np.float32)
        # �ۿۻر�
        discounted_reward = np.zeros((self.idx, ), dtype=np.float32)

        for step in reversed(range(self.idx)):
            # �����ۿ۽���
            last_reward = self.rewards[step] + self.gamma * last_reward
            # �洢
            discounted_reward[step] = last_reward

        return discounted_reward

    def __len__(self):
        """
        ��ȡ�켣����
        :return:
        """
        return self.idx


class SimpleReplayBuffer(BaseBuffer):
    """
    ����طŻ�����
    """

    def __init__(self,
                 state_dim: tuple | int,
                 action_num: int,
                 capacity: int,
                 batch_size: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        ���캯��
        :param state_dim: ״̬ά��
        :param action_num: ��������
        :param capacity: ��������С
        :param batch_size: ����С
        :param device: �����豸
        """
        super(SimpleReplayBuffer, self).__init__(device)

        assert capacity > batch_size, f"����������Ҫ���ڲ���������С"

        # �����ṹ
        self.capacity = capacity
        self.batch_size = batch_size

        # �ڴ�
        if isinstance(state_dim, int):
            state_dim = (state_dim, )
        self.states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_num), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, ), dtype=np.float32)
        self.dones = np.zeros((self.capacity, ), dtype=np.float32)

        # ��ǰ��ŵ�λ��
        self.idx = 0
        # �Ѿ�ʹ�ó���
        self.length = 0

    def store(self,
              state: np.ndarray,
              action: Union[np.ndarray, int],
              next_state: np.ndarray,
              reward: float,
              done: bool) -> None:
        """
        �洢����
        :param state: ��ǰ״̬
        :param action: ����
        :param next_state: ��һ��״̬
        :param reward: ����
        :param done: �Ƿ����
        :return:
        """
        # �洢
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        # ָ��λ�ü�1
        self.idx += 1
        self.idx = self.idx % self.capacity

        # �Ѿ�ʹ�ó��ȼ�1
        self.length += 1
        if self.length >= self.capacity:
            self.length = self.capacity

    def sample(self) -> tuple:
        """
        ����
        :return:
        """
        # ����������
        index = np.random.choice(self.length, size=self.batch_size, replace=False)

        # ��Ϊ����
        state = self.to_tensor(self.states[index])
        action = self.to_tensor(self.actions[index])
        next_state = self.to_tensor(self.next_states[index])
        reward = self.to_tensor(self.rewards[index]).unsqueeze(-1)
        done = self.to_tensor(self.dones[index]).unsqueeze(-1)

        return state, action, next_state, reward, done

    def __len__(self) -> int:
        """
        ��ȡ��ǰ��������ʹ�ó���
        :return:
        """
        return self.length


class RolloutBuffer(BaseBuffer):
    """
    ���л����켣���ɳ���������
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
        ���캯��
        :param state_dim: ״̬ά��
        :param action_num: ��������
        :param env: ���л���
        :param capacity: ����������
        :param batch_size: ��������С
        :param gamma: �����ۿ�ϵ��
        :param gae_lambda: GAE�ۿ�ϵ��
        :param device: �����豸
        """
        super(RolloutBuffer, self).__init__(device)

        # ��������
        self.state_dim = state_dim if isinstance(state_dim, tuple) else (state_dim, )
        self.action_num = action_num
        self.num_envs = env.num_envs
        self.env = env
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # �Ƿ�����
        self.full = False
        # ��ǰλ��
        self.idx = 0
        # ��¼�켣ģ�������ʱ���״̬�ͽ�����ʶ
        self.last_state = None
        self.last_done = None

        # �ڴ�
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
        �������
        :return:
        """
        # �������ݹ���
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self.dones.fill(0)

        # ���ñ�ʶ��
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
        �洢����
        :param state:
        :param action:
        :param reward:
        :param log_prob:
        :param value:
        :param done:
        :return:
        """
        if not self.full:
            # ����ά��
            if len(action.shape) == 1:
                action = np.expand_dims(action, -1)
            if len(log_prob.shape) == 2:
                log_prob = np.squeeze(log_prob, -1)
            if len(value.shape) == 2:
                value = np.squeeze(value, -1)
            # �洢
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward
            self.log_probs[self.idx] = log_prob
            self.values[self.idx] = value
            self.dones[self.idx] = done
            # ָ�����
            self.idx += 1

            # �Ƿ�洢��
            if self.idx >= self.capacity:
                # �洢���˱�־λ
                self.full = True

    def compute_returns_and_advantage(self, v_next: torch.Tensor):
        """
        ����GAE���ۿۻر�
        :return:
        """
        # ��һʱ�̵�Vֵ
        v_next = v_next.squeeze().detach().cpu().numpy()
        # ��һʱ�̵ĵ����Ծ�ֵ
        last_gae_lam = np.zeros((self.num_envs, ), dtype=np.float32)
        # ��һʱ�̵Ľ���
        last_reward = v_next

        for step in reversed(range(self.capacity)):
            # ����������ʶ
            non_terminal = 1.0 - self.dones[step]
            # ��һʱ��״ֵ̬
            if step == self.capacity - 1:
                next_values = v_next
            else:
                next_values = self.values[step + 1]
            # ���㵥���Ծ�
            delta = self.rewards[step] + self.gamma * next_values * non_terminal - self.values[step]
            # ����GAE
            last_gae_lam = delta + self.gamma * self.gae_lambda * non_terminal * last_gae_lam
            # �����ۿ۽���
            last_reward = self.rewards[step] + self.gamma * non_terminal * last_reward
            # �洢
            self.advantages[step] = last_gae_lam
            # �����ۿۻر�
            self.returns[step] = last_reward

    def roll_out(self,
                 agent,
                 seed: int = None,
                 ):
        """
        �켣ģ��
        :param agent: ������
        :param seed: ��������
        :return:
        """
        # ����һ��ģ�������λ�ÿ�ʼ
        state = self.last_state
        done = self.last_done

        # �����������û���
        if state is None and done is None:
            state, _ = self.env.reset()
            done = np.array([False for _ in range(self.num_envs)], dtype=bool)

        while not self.full:
            # ѡ����
            action, log_pa, _ = agent.select_action(state, "train")
            # ����״ֵ̬
            value = agent.value_model(self.to_tensor(state))
            # ����
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # ������ʶ
            done = np.logical_or(terminated, truncated)
            # �洢
            self.store(state, action, reward, log_pa.detach().cpu().numpy(),
                       value.detach().cpu().numpy(), done.astype(np.float32))
            # ����״̬
            state = next_state.copy()

            # ���û���
            if np.any(done):
                state, _ = self.env.reset(seed=seed, options={"reset_mask": done})

        # ����״ֵ̬
        next_value = agent.value_model(self.to_tensor(state))

        # �����ۿۻر���GAE
        self.compute_returns_and_advantage(next_value)

        # ��¼ģ�����λ��
        self.last_state = state.copy()
        self.last_done = done.copy()

    def sample(self, total: bool = False):
        """
        �켣����
        :param total: �Ƿ�������й켣
        :return:
        """
        # ��������
        batch_size = self.batch_size if not total else self.capacity * self.num_envs

        # չƽ������ת��Ϊ����
        state = self.to_tensor(self.swap_and_flatten(self.states))
        action = self.to_tensor(self.swap_and_flatten(self.actions))
        log_prob = self.to_tensor(self.swap_and_flatten(self.log_probs))
        value = self.to_tensor(self.swap_and_flatten(self.values))
        advantage = self.to_tensor(self.swap_and_flatten(self.advantages))
        returns = self.to_tensor(self.swap_and_flatten(self.returns))

        # ��ʼ����
        start_idx = 0

        # ��������ѡ������
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
    ���Ⱦ���طŻ�����
    """
    def __init__(self,
                 capacity: int,
                 batch_size: int,
                 alpha: float,
                 beta: float,
                 beta_inc_step: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        ���캯��
        :param capacity:
        :param batch_size:
        :param alpha:
        :param beta:
        :param beta_inc_step:
        :param device:
        """
        super(PrioritizedReplayBuffer, self).__init__(device)

        assert 0 <= alpha <= 1, f"alpha Ҫ��[0, 1]�����ڣ�ʵ��Ϊ{alpha}"
        assert 0 <= beta < 1, f"beta Ҫ��[0, 1)�����ڣ�ʵ��Ϊ{beta}"

        # �洢�������
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_inc_step = beta_inc_step
        # beta�б�
        self.beta_list = np.linspace(beta, 1.0, beta_inc_step)
        # ��ǰ����
        self.length = 0

        # �ڴ�
        self.memory = SumTree(capacity)

    def _get_priority(self, td_error):
        """
        �������ȼ�
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
        �洢����
        :param td_error: TD���
        :param state: ��ǰ״̬
        :param action: ����
        :param next_state: ��һ��״̬
        :param reward: ����
        :param done: �Ƿ����
        :return:
        """
        # �������ȼ�
        p = self._get_priority(td_error)
        # �������ȼ��洢
        self.memory.add(p, (state, action, next_state, reward, done))

    def sample(self, size: int = None):
        """
        ����
        :param size: ��������С
        :return:
        """
        batched_data = []
        index = []
        priorities = []

        # ������С
        bs = size if size is not None else self.batch_size

        segment = self.memory.total_priority / bs

        for i in range(bs):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            # ��ȡҶ�ӽڵ�
            idx, p, data = self.memory.get_leaf(v)
            batched_data.append(data)
            index.append(idx)
            priorities.append(p)

        # �����������
        sampling_probabilities = np.array(priorities) / self.memory.total_priority
        # ����Ȩ��
        weights = (len(self.memory.data) * sampling_probabilities) ** (-self.beta)
        # Ȩ�ع�һ��
        weights /= weights.max()
        # ת��Ϊtensor
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # �ۺ�ΪTensor����
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

        # ���²���
        self.length += 1
        # ����beta
        if self.length > len(self.beta_list):
            self.beta = 1.0
        else:
            self.beta = self.beta_list[self.length]

        return index, weights, state, action, next_state, reward, done

    def update(self, idx, td_errors):
        for idx, td_error in zip(idx, td_errors):
            # �������ȼ�
            p = self._get_priority(td_error)
            # �������ȼ�
            self.memory.update(idx, p)

    def __len__(self):
        """
        ��ȡ����������
        :return:
        """
        return self.length

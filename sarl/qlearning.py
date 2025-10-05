# -*- coding:gbk -*-

"""
Q学习智能体
"""


import utils
import torch
import net
from . import agent

__all__ = ["QLearningAgent"]


class QLearningAgent(agent.ValueBasedAgent):
    """
    Q学习智能体
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 target_model: net.PolicyModel,
                 selector: utils.selector.Selector,
                 q_type: str,
                 gamma: float = 0.99,
                 lr: float = 0.001,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0):
        """
        构造函数
        :param policy_model: 在线网络
        :param target_model: 目标网络
        :param selector: 动作选择器
        :param q_type: Q学习类型, DQN, Double DQN
        :param gamma: 折扣系数
        :param lr: 学习率
        :param tau: 网络更新参数
        :param update_interval: 网络更新步长
        """
        super().__init__(policy_model, target_model, selector, gamma, lr, tau, update_interval, grad_norm_clipping)

        self.q_type = q_type

    def calculate_td_target(self, next_state: torch.Tensor, reward, done):
        """
        计算TD目标
        :param next_state: 下一时刻状态
        :param reward: 奖励
        :param done: 结束符号
        :return:
        """
        if self.q_type == 'dqn':
            return self._dqn_td_target(next_state, reward, done)
        elif self.q_type == 'ddqn':
            return self._ddqn_td_target(next_state, reward, done)

        return None

    def _dqn_td_target(self, next_state: torch.Tensor, reward, done) -> torch.Tensor:
        """
        Q学习计算TD目标
        :param next_state: 下一状态
        :param reward: 奖励
        :param done: 结束标识
        :return:
        """
        max_a_q_sp = torch.max(self.target_model(next_state), dim=-1, keepdim=True)[0]
        target_q_sa = reward + (self.gamma * max_a_q_sp * (1 - done))

        return target_q_sa

    def _ddqn_td_target(self, next_state: torch.Tensor, reward, done) -> torch.Tensor:
        """
        双重Q学习计算TD目标
        :param next_state: 下一状态
        :param reward: 奖励
        :param done: 结束标识
        :return:
        """
        # 批大小
        batch_size = reward.size(0)

        argmax_a_q_sp = torch.argmax(self.policy_model(next_state), dim=-1)
        q_sp = self.target_model(next_state)
        max_a_q_sp = q_sp[torch.arange(batch_size), argmax_a_q_sp].unsqueeze(-1)
        target_q_sa = reward + (self.gamma * max_a_q_sp * (1 - done))

        return target_q_sa

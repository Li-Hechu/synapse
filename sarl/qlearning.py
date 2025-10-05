# -*- coding:gbk -*-

"""
Qѧϰ������
"""


import utils
import torch
import net
from . import agent

__all__ = ["QLearningAgent"]


class QLearningAgent(agent.ValueBasedAgent):
    """
    Qѧϰ������
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
        ���캯��
        :param policy_model: ��������
        :param target_model: Ŀ������
        :param selector: ����ѡ����
        :param q_type: Qѧϰ����, DQN, Double DQN
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: ������²���
        :param update_interval: ������²���
        """
        super().__init__(policy_model, target_model, selector, gamma, lr, tau, update_interval, grad_norm_clipping)

        self.q_type = q_type

    def calculate_td_target(self, next_state: torch.Tensor, reward, done):
        """
        ����TDĿ��
        :param next_state: ��һʱ��״̬
        :param reward: ����
        :param done: ��������
        :return:
        """
        if self.q_type == 'dqn':
            return self._dqn_td_target(next_state, reward, done)
        elif self.q_type == 'ddqn':
            return self._ddqn_td_target(next_state, reward, done)

        return None

    def _dqn_td_target(self, next_state: torch.Tensor, reward, done) -> torch.Tensor:
        """
        Qѧϰ����TDĿ��
        :param next_state: ��һ״̬
        :param reward: ����
        :param done: ������ʶ
        :return:
        """
        max_a_q_sp = torch.max(self.target_model(next_state), dim=-1, keepdim=True)[0]
        target_q_sa = reward + (self.gamma * max_a_q_sp * (1 - done))

        return target_q_sa

    def _ddqn_td_target(self, next_state: torch.Tensor, reward, done) -> torch.Tensor:
        """
        ˫��Qѧϰ����TDĿ��
        :param next_state: ��һ״̬
        :param reward: ����
        :param done: ������ʶ
        :return:
        """
        # ����С
        batch_size = reward.size(0)

        argmax_a_q_sp = torch.argmax(self.policy_model(next_state), dim=-1)
        q_sp = self.target_model(next_state)
        max_a_q_sp = q_sp[torch.arange(batch_size), argmax_a_q_sp].unsqueeze(-1)
        target_q_sa = reward + (self.gamma * max_a_q_sp * (1 - done))

        return target_q_sa

# -*- coding:gbk -*-

"""
���ڲ����ݶȵ�������
"""

import torch
from . import agent
import utils
import net

__all__ = ["REINFORCEAgent",
           "VPGAgent",
           "A3CAgent",
           "A2CAgent",
           "PPOAgent"]


class REINFORCEAgent(agent.PGBasedAgent):
    """
    REINFORCE������
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 grad_norm_clipping: float = 1.0,
                 ):
        """
        ���캯��
        :param policy_model: ��������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        """
        super(REINFORCEAgent, self).__init__(policy_model, None, gamma, lr, grad_norm_clipping,
                                             False, None, None)

    def optimize(self, trajectory):
        """
        �Ż�������
        :param trajectory: ������켣
        :return:
        """
        # ���
        states, actions, returns = trajectory
        # ��������
        log_pa, _ = self.policy_model.evaluate_actions(states, actions)

        # ���ƺ���
        advantage = returns
        # ���������ʧ
        policy_loss = -(advantage.detach() * log_pa).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.policy_optimizer.step()


class VPGAgent(agent.PGBasedAgent):
    """
    VPG������
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 value_model: net.ValueModel,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 grad_norm_clipping: float = 1.0,
                 entropy_loss_weight: float = 0.01
                 ):
        """
        ���캯��
        :param policy_model: ��������
        :param value_model: ��ֵ����
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param entropy_loss_weight: ����ʧȨ��
        """
        super(VPGAgent, self).__init__(policy_model, value_model, gamma, lr, grad_norm_clipping, False, None,
                                       entropy_loss_weight)

    def optimize(self, trajectory: tuple):
        """
        �Ż�������
        :param trajectory: ������켣
        :return:
        """
        # ���
        states, actions, returns = trajectory
        # ��������
        log_pa, entropy = self.policy_model.evaluate_actions(states, actions)
        # ״ֵ̬
        values = self.value_model(states)

        # �������ƺ���
        advantage = returns.detach() - values
        # ���������ʧ
        policy_loss = -(advantage.detach() * log_pa).mean() - self.entropy_loss_weight * entropy.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.policy_optimizer.step()

        # �����ֵ��ʧ
        value_error = advantage
        value_loss = value_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
        self.value_optimizer.step()


class A3CAgent(agent.PGBasedAgent):
    """
    A3C������
    """

    def __init__(self,
                 shared_policy_model: net.PolicyModel,
                 shared_value_model: net.ValueModel,
                 shared_policy_optimizer,
                 shared_value_optimizer,
                 policy_model: net.PolicyModel,
                 value_model: net.ValueModel,
                 gamma: float = 0.99,
                 grad_norm_clipping: float = 1.0,
                 entropy_loss_weight: float = 0.01
                 ):
        """
        ���캯��
        :param shared_policy_model: �����������
        :param shared_value_model: �����ֵ����
        :param shared_policy_optimizer: ������������Ż���
        :param shared_value_optimizer: �����ֵ�����Ż���
        :param policy_model: ���ز�������
        :param value_model: ���ؼ�ֵ����
        :param gamma: �ۿ�ϵ��
        :param grad_norm_clipping:  �ݶȲü���Χ
        :param entropy_loss_weight: ����ʧȨ��
        """
        super(A3CAgent, self).__init__(policy_model, value_model, gamma, 0, grad_norm_clipping, False, None,
                                       entropy_loss_weight)

        # ����ģ��
        self.shared_policy_model = shared_policy_model
        self.shared_value_model = shared_value_model
        # �����Ż���
        self.policy_optimizer = shared_policy_optimizer
        self.value_optimizer = shared_value_optimizer

        # ����ͬ��
        self.policy_model.load_state_dict(self.shared_policy_model.state_dict())
        self.value_model.load_state_dict(self.shared_value_model.state_dict())

    def optimize(self, trajectory):
        """
        �Ż�������
        :param trajectory: ������켣
        :return:
        """
        # ���
        states, actions, returns = trajectory
        # ��������
        log_pa, entropy = self.policy_model.evaluate_actions(states, actions)
        # ״ֵ̬
        values = self.value_model(states)

        # �������ƺ���
        advantage = returns.detach() - values
        # ���������ʧ
        policy_loss = -(advantage.detach() * log_pa).mean() - self.entropy_loss_weight * entropy.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        # ����ģ�͵��ݶȸ��Ƶ�����ģ��
        for param, shared_param in zip(self.policy_model.parameters(), self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.policy_optimizer.step()
        # ͬ������
        self.policy_model.load_state_dict(self.shared_policy_model.state_dict())

        # �����ֵ��ʧ
        value_error = advantage
        value_loss = value_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
        # ����ģ�͵��ݶȸ��Ƶ�����ģ��
        for param, shared_param in zip(self.value_model.parameters(), self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.value_optimizer.step()
        # ͬ������
        self.value_model.load_state_dict(self.shared_value_model.state_dict())


class A2CAgent(agent.PGBasedAgent):
    """
    A2C������
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 value_model: net.ValueModel,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 grad_norm_clipping: float = 1.0,
                 gae_lambda: float = 0.95,
                 entropy_loss_weight: float = 0.01,
                 gae_norm: bool = True
                 ):
        """
        ���캯��
        :param policy_model: ��������
        :param value_model: ��ֵ����
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param gae_lambda: GAE�Ľ����ۿ�ϵ��
        :param entropy_loss_weight: ����ʧȨ��
        :param gae_norm: �Ƿ��׼��GAE
        """
        super(A2CAgent, self).__init__(policy_model, value_model, gamma, lr, grad_norm_clipping, True, gae_lambda,
                                       entropy_loss_weight)

        # �Ƿ��׼��GAE
        self.gae_norm = gae_norm

    def optimize(self,
                 buffer: utils.buffer.RolloutBuffer,
                 seed=None):
        """
        �Ż�ģ��
        :return:
        """
        # �켣ģ��
        buffer.roll_out(self, seed)

        # һ��������������ݣ�ֻѭ��һ��
        for trajectory in buffer.sample(total=True):
            # ���
            states, actions, _, _, advantages, returns = trajectory
            # �Ե�ǰ���Խ�������
            log_prob, entropy = self.policy_model.evaluate_actions(states, actions)
            # �����״ֵ̬
            values = self.value_model(states)

            # ��׼��GAE
            if self.gae_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # ������ʧ
            policy_loss = -(advantages.detach() * log_prob).mean() - self.entropy_loss_weight * entropy.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
            self.policy_optimizer.step()

            # ��ֵ��ʧ
            value_loss = (returns.detach() - values).pow(2).mul(0.5).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
            self.value_optimizer.step()

        # ���������
        buffer.clear()


class PPOAgent(A2CAgent):
    """
    PPO������
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 value_model: net.ValueModel,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 grad_norm_clipping: float = 1.0,
                 gae_lambda: float = 0.95,
                 entropy_loss_weight: float = 0.01,
                 gae_norm: bool = True,
                 update_epoch: int = 5,
                 policy_clipping: float = 0.2,
                 value_clipping: float = 0.2,
                 ):
        """
        ���캯��
        :param policy_model: ��������
        :param value_model: ��ֵ����
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param gae_lambda: GAE�Ľ����ۿ�ϵ��
        :param entropy_loss_weight: ����ʧȨ��
        :param gae_norm: �Ƿ��׼��GAE
        :param update_epoch: �����ظ��Ż�����
        :param policy_clipping: �����¾ɸ��ʱȲü���Χ
        :param value_clipping: �¾ɼ�ֵ֮��ü���Χ
        """
        super(PPOAgent, self).__init__(policy_model, value_model, gamma, lr, grad_norm_clipping, gae_lambda,
                                       entropy_loss_weight, gae_norm)

        assert 0 <= policy_clipping, f"���Բü���Χ����Ӧ���ڵ���0��ʵ��Ϊ {policy_clipping}"
        assert 0 <= value_clipping, f"��ֵ�ü���Χ����Ӧ���ڵ���0��ʵ��Ϊ {value_clipping}"

        self.update_epoch = update_epoch
        self.policy_clipping = policy_clipping
        self.value_clipping = value_clipping

    def optimize(self,
                 buffer: utils.buffer.RolloutBuffer,
                 seed=None
                 ):
        """
        �Ż�������
        :param buffer: �켣���ɳ���������
        :param seed: �����������
        :return:
        """
        # ģ��켣
        buffer.roll_out(self, seed)

        # ������´���
        for _ in range(self.update_epoch):
            # �������зֳ�mini-batch�������θ�������
            for trajectory in buffer.sample(total=False):
                # ����
                states, actions, log_prob, values, advantages, returns = trajectory

                # ��GAE��׼��
                if self.gae_norm:
                    advantages = (advantages - torch.mean(advantages).item()) / (torch.std(advantages).item() + 1e-6)

                # �²����µĶ����������ʺ���
                new_log_prob, new_entropy = self.policy_model.evaluate_actions(states, actions)
                # �¼�ֵ�����µ�״ֵ̬
                new_values = self.value_model(states)

                # �¾ɲ��Ը��ʱ�
                ratio = torch.exp(new_log_prob - log_prob.detach())
                # ԭʼ�Ż�Ŀ��
                pi_obj = ratio * advantages.detach()
                # �ü�����Ż�Ŀ��
                pi_obj_clipped = torch.clip(ratio, 1.0 - self.policy_clipping,
                                            1.0 + self.policy_clipping) * advantages.detach()
                # ������ʧ����
                policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean() - self.entropy_loss_weight * new_entropy.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
                self.policy_optimizer.step()

                # ���µ�״ֵ̬���вü�
                new_value_clipped = values.detach() + torch.clip((new_values - values.detach()),
                                                                 -self.value_clipping, self.value_clipping)
                # ԭʼ�ļ�ֵ��ʧ
                v_obj = (returns.detach() - new_values).pow(2)
                # �ü���ļ�ֵ��ʧ
                v_obj_clipped = (returns.detach() - new_value_clipped).pow(2)
                # ��ֵ��ʧ
                value_loss = torch.max(v_obj, v_obj_clipped).mul(0.5).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
                self.value_optimizer.step()

        # ����켣���ɻ�����������
        buffer.clear()

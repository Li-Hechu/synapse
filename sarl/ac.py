# -*- coding:gbk -*-

"""
Actor-Critic���������
"""

import numpy as np
import torch
from . import agent
import utils
import net

__all__ = ["DDPGAgent",
           "TD3Agent",
           "SACAgent"]


class DDPGAgent(agent.ACBasedAgent):
    """
    DDPG������
    """
    def __init__(self,
                 online_policy_model: net.PolicyModel,
                 target_policy_model: net.PolicyModel,
                 online_value_model: net.ValueModel,
                 target_value_model: net.ValueModel,
                 selector: utils.selector.GaussianSelector,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0
                 ):
        """
        ���캯��
        :param online_policy_model: ���߲�������
        :param target_policy_model: Ŀ���������
        :param online_value_model: ���߼�ֵ����
        :param target_value_model: Ŀ���ֵ����
        :param selector: ����ѡ����
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: ������²���
        :param update_interval: ������¼��
        :param grad_norm_clipping: �ݶȲü���Χ
        """
        super(DDPGAgent, self).__init__(gamma, lr, tau, update_interval, grad_norm_clipping)

        self.policy_model = online_policy_model
        self.target_policy_model = target_policy_model
        self.online_value_model = online_value_model
        self.target_value_model = target_value_model

        # ����ѡ����
        self.selector = selector

        # ͬ��Ȩ��
        self.target_value_model.load_state_dict(self.online_value_model.state_dict())
        self.target_policy_model.load_state_dict(self.policy_model.state_dict())

        # �Ż���
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.online_value_model.parameters(), lr=self.lr)

        # �����豸
        self.device: torch.device = self.policy_model.device

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        ѡ����
        :param state: ��ǰ״̬
        :param a_type: ����ѡ������
        :return:
        """
        # ״̬ת��Ϊ����
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        if a_type == "eval":
            with torch.no_grad():
                q_val = self.policy_model(state).detach().squeeze().cpu().numpy()
            return q_val
        elif a_type == "train":
            return self.selector.select_action(state)

        return None

    def optimize(self, experience):
        """
        ����������
        :param experience: ����Ԫ��
        :return:
        """
        # ���
        state, action, next_state, reward, done = experience

        # -----------��ֵ�������---------
        argmax_a_q_sp = self.target_policy_model(next_state)
        max_a_q_sp = self.target_value_model(next_state, argmax_a_q_sp)
        # ����tdĿ��
        target_q_sa = reward + self.gamma * max_a_q_sp * (1 - done)
        # ����y
        q_sa = self.online_value_model(state, action)
        # td���
        td_error = q_sa - target_q_sa.detach()
        # ��ֵ��ʧ
        value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.grad_norm_clipping)
        self.value_optimizer.step()

        # -----------�����������-----------
        argmax_a_q_s = self.policy_model(state)
        max_a_q_s = self.online_value_model(state, argmax_a_q_s)
        # ������ʧ
        policy_loss = -max_a_q_s.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.policy_optimizer.step()

        # -----------�����������----------
        # ������1
        self.step += 1
        # �����������
        if self.step % self.update_interval == 0:
            self.update_network()

    def update_network(self) -> None:
        """
        �����������
        :return:
        """
        self.mix_network(self.target_value_model, self.online_value_model, self.tau)
        self.mix_network(self.target_policy_model, self.policy_model, self.tau)


class TD3Agent(DDPGAgent):
    """
    TD3������
    """

    def __init__(self,
                 online_policy_model: net.PolicyModel,
                 target_policy_model: net.PolicyModel,
                 online_value_model: net.ValueModel,
                 target_value_model: net.ValueModel,
                 online_value_model_1: net.ValueModel,
                 target_value_model_1: net.ValueModel,
                 selector: utils.selector.GaussianSelector,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 tau: float = 0.1,
                 update_interval: int = 4,
                 grad_norm_clipping: float = 1.0,
                 target_noise_clipping: float = 1.0,
                 noise_sigma: float = 1.0
                 ):
        """
        ���캯��
        :param online_policy_model: ���߲�������
        :param target_policy_model: Ŀ���������
        :param online_value_model: ���߼�ֵ����
        :param target_value_model: Ŀ���ֵ����
        :param online_value_model_1: ���߼�ֵ����1
        :param target_value_model_1: Ŀ���ֵ����1
        :param selector: ����ѡ����
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: ������±���
        :param update_interval: ���¼��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param target_noise_clipping: �����ü���Χ
        :param noise_sigma: ������׼��
        """
        super(TD3Agent, self).__init__(online_policy_model,
                                       target_policy_model,
                                       online_value_model,
                                       target_value_model,
                                       selector, gamma, lr, tau, update_interval, grad_norm_clipping)

        assert target_noise_clipping > 0.0, "Ŀ�������ü���ΧӦ�ô���0"
        assert noise_sigma >= 0.0, "��׼��Ӧ�����ڵ���0"

        # ������Χ
        self.target_noise_clipping = target_noise_clipping
        # ������׼��
        self.noise_sigma = noise_sigma

        # TD3����һ���ֵ����
        self.online_value_model_1 = online_value_model_1
        self.target_value_model_1 = target_value_model_1
        # ͬ��Ȩ��
        self.target_value_model_1.load_state_dict(self.online_value_model_1.state_dict())
        # ������ֵ������Ż���
        self.value_optimizer_1 = torch.optim.Adam(self.online_value_model_1.parameters(), lr=self.lr)

    def optimize(self, experience):
        """
        ����������
        :param experience: ����Ԫ��
        :return:
        """
        # ���
        state, action, next_state, reward, done = experience

        # -----------��ֵ�������---------
        # ����
        noise = torch.normal(mean=torch.zeros_like(action), std=self.noise_sigma)
        noise = torch.clamp(noise, -self.target_noise_clipping, self.target_noise_clipping)
        # ��Ŀ�궯��������
        argmax_a_q_sp = self.target_policy_model(next_state)
        noisy_action = argmax_a_q_sp + noise
        noisy_action = torch.clamp(noisy_action, self.policy_model.action_min, self.policy_model.action_max)

        max_a_q_sp_a = self.target_value_model(next_state, noisy_action)
        max_a_q_sp_b = self.target_value_model_1(next_state, noisy_action)
        # ȡ��Сֵ
        max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)

        # ����tdĿ��
        td_target = reward + self.gamma * max_a_q_sp * (1 - done)

        q_sa_a = self.online_value_model(state, action)
        q_sa_b = self.online_value_model_1(state, action)
        # ����TD���
        td_error_a = td_target.detach() - q_sa_a
        td_error_b = td_target.detach() - q_sa_b
        # ��ֵ��ʧ
        value_loss_a = td_error_a.pow(2).mul(0.5).mean()
        value_loss_b = td_error_b.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.grad_norm_clipping)
        self.value_optimizer.step()

        self.value_optimizer_1.zero_grad()
        value_loss_b.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_1.parameters(), self.grad_norm_clipping)
        self.value_optimizer_1.step()

        # ������1
        self.step += 1
        if self.step % self.update_interval == 0:
            # ----------�����������---------
            # ���߶���
            argmax_a_q_s = self.policy_model(state)
            # �������ֵ
            max_a_q_s = self.online_value_model(state, argmax_a_q_s)
            # ������ʧ
            policy_loss = -max_a_q_s.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
            self.policy_optimizer.step()

            # --------Ŀ�������������--------
            self.update_network()

    def update_network(self):
        """
        ��������
        :return:
        """
        self.mix_network(self.target_value_model, self.online_value_model, self.tau)
        self.mix_network(self.target_value_model_1, self.online_value_model_1, self.tau)
        self.mix_network(self.target_policy_model, self.policy_model, self.tau)


class SACAgent(agent.ACBasedAgent):
    """
    SAC������
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 online_value_model_a: net.ValueModel,
                 target_value_model_a: net.ValueModel,
                 online_value_model_b: net.ValueModel,
                 target_value_model_b: net.ValueModel,
                 target_entropy: float,
                 entropy_lr: float = 0.0001,
                 gamma: float = 0.99,
                 lr: float = 0.001,
                 tau: float = 0.1,
                 update_interval: int = 4,
                 grad_norm_clipping: float = 1.0
                 ):
        """
        ���캯��
        :param policy_model: ��������
        :param online_value_model_a: ���߼�ֵ����a
        :param target_value_model_a: Ŀ���ֵ����b
        :param online_value_model_b: ���߼�ֵ����a
        :param target_value_model_b: Ŀ���ֵ����b
        :param target_entropy: Ŀ����
        :param entropy_lr: ��ѧϰ��
        :param gamma: �ۿ�ϵ��
        :param lr: ����ѧϰ��
        :param tau: Ŀ�����������
        :param update_interval: ���¼��
        :param grad_norm_clipping: �ݶȲü���Χ
        """
        super(SACAgent, self).__init__(gamma, lr, tau, update_interval, grad_norm_clipping)

        # Ŀ��������ʽ
        self.target_entropy = target_entropy
        # ��ѧϰ��
        self.entropy_lr = entropy_lr

        self.online_value_model_a = online_value_model_a
        self.target_value_model_a = target_value_model_a
        self.online_value_model_b = online_value_model_b
        self.target_value_model_b = target_value_model_b
        self.policy_model = policy_model
        # ͬ��Ȩ��
        self.target_value_model_a.load_state_dict(self.online_value_model_a.state_dict())
        self.target_value_model_b.load_state_dict(self.online_value_model_b.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # �Ż���
        self.value_optimizer_a = torch.optim.Adam(self.online_value_model_a.parameters(), lr=self.lr)
        self.value_optimizer_b = torch.optim.Adam(self.online_value_model_b.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.entropy_lr)

        # �����豸
        self.device: torch.device = self.policy_model.device

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        ѡ����
        :param state: ��ǰ״̬
        :param a_type: ����ѡ������
        :return:
        """
        # ״̬ת��Ϊ����
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        with torch.no_grad():
            sample_action, _, _, determined_action = self.policy_model.full_pass(state)

        if a_type == "train":
            return sample_action.squeeze().detach().cpu().numpy()
        elif a_type == "eval":
            return determined_action.squeeze().detach().cpu().numpy()

        return None

    def optimize(self, experience):
        """
        ����������
        :param experience: ����Ԫ��
        :return:
        """
        # ���
        state, action, next_state, reward, done = experience

        # ---------alpha��ʧ----------
        _, log_pi_s, _, _ = self.policy_model.full_pass(state)
        target_alpha = (log_pi_s + self.target_entropy).detach()
        alpha_loss = -(self.log_alpha * target_alpha).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        # --------------��ֵ��ʧ--------------
        # ��������
        a_next, a_next_log_prob, _, _ = self.policy_model.full_pass(next_state)
        # ����V
        q_spap_a = self.target_value_model_a(next_state, a_next)
        q_spap_b = self.target_value_model_b(next_state, a_next)
        q_spap = torch.min(q_spap_a, q_spap_b) - alpha * a_next_log_prob
        # tdĿ��
        td_target = reward + self.gamma * q_spap * (1.0 - done)
        # q_sa
        q_sa_a = self.online_value_model_a(state, action)
        q_sa_b = self.online_value_model_b(state, action)
        # ��ֵ��ʧ
        value_loss_a = (td_target.detach() - q_sa_a).pow(2).mul(0.5).mean()
        value_loss_b = (td_target.detach() - q_sa_b).pow(2).mul(0.5).mean()

        self.value_optimizer_a.zero_grad()
        value_loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_a.parameters(), self.grad_norm_clipping)
        self.value_optimizer_a.step()

        self.value_optimizer_b.zero_grad()
        value_loss_b.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_b.parameters(), self.grad_norm_clipping)
        self.value_optimizer_b.step()

        # ------------������ʧ------------
        a_cur, a_cur_log_prob, _, _ = self.policy_model.full_pass(state)
        cur_q_sa_a = self.online_value_model_a(state, a_cur)
        cur_q_sa_b = self.online_value_model_b(state, a_cur)
        cur_q_sa = torch.min(cur_q_sa_a, cur_q_sa_b)
        # ������ʧ
        policy_loss = (alpha * a_cur_log_prob - cur_q_sa).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.policy_optimizer.step()

        # -----------�����������----------
        # ������1
        self.step += 1
        # �����������
        if self.step % self.update_interval == 0:
            self.update_network()

    def update_network(self):
        """
        ��������
        :return:
        """
        self.mix_network(self.target_value_model_a, self.online_value_model_a, self.tau)
        self.mix_network(self.target_value_model_b, self.online_value_model_b, self.tau)



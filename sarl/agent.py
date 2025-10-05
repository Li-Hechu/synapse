# -*- coding:gbk -*-

"""
ǿ��ѧϰ��������
"""

import torch
import numpy as np
import net
import utils

__all__ = ["Agent",
           "ValueBasedAgent",
           "PGBasedAgent",
           "ACBasedAgent"]


class Agent:
    """
    �����������
    """

    def __init__(self,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 tau: float | None = 0.1,
                 update_interval: int | None = 5,
                 grad_norm_clipping: float = 1.0
                 ):
        """
        ���캯��
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: ���������
        :param grad_norm_clipping: �ݶȲü����ֵ
        """
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_interval = update_interval
        self.grad_norm_clipping = grad_norm_clipping

        # ��ǰ����
        self.step = 0
        # ����ѡ����
        self.selector = None
        # ��������
        self.policy_model = None
        # �����豸
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        ѡ����
        :param state: ��ǰ״̬
        :param a_type: ����ѡ������, train, eval
        :return:
        """
        ...

    def optimize(self, *args, **kwargs):
        """
        ���������
        :return:
        """
        ...

    def update_network(self) -> None:
        """
        ����Ŀ������
        :return:
        """
        ...

    def save_policy(self, filename) -> None:
        """
        �������
        :param filename: ����·��
        :return:
        """
        torch.save(self.policy_model.state_dict(), filename)

    def load_policy(self, filename) -> None:
        """
        ���ز���
        :param filename: ����·��
        :return:
        """
        self.policy_model.load_state_dict(torch.load(filename, weights_only=True))

    @staticmethod
    def mix_network(target_network: torch.nn.Module, online_network: torch.nn.Module, tau: float):
        """
        �������Ȩ��
        :param target_network: Ŀ������
        :param online_network: ��������
        :param tau: ���Ȩ��
        :return:
        """
        for target, online in zip(target_network.parameters(), online_network.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)


class ValueBasedAgent(Agent):
    """
    ���ڼ�ֵ�������壺DQN, Double DQN������
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 target_model: net.PolicyModel,
                 selector: utils.Selector,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0
                 ):
        """
        ���캯��
        """
        super(ValueBasedAgent, self).__init__(gamma, lr, tau, update_interval, grad_norm_clipping)

        # ����ѡ����
        self.selector = selector

        # Ŀ���������������
        self.policy_model = policy_model
        self.target_model = target_model
        # ͬ��Ȩ��
        self.target_model.load_state_dict(self.policy_model.state_dict())
        # �Ż���
        self.value_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)

        # �����豸
        self.device: torch.device = self.policy_model.device

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        ѡ����
        :param state: ��ǰ״̬
        :param a_type: ����ѡ������, train, eval
        :return:
        """
        # ״̬ת��Ϊ����
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # ����ģʽѡ����������̰���㷨
        if a_type == "eval":
            with torch.no_grad():
                q_val = self.policy_model(state).detach().squeeze().cpu().numpy()
            return np.argmax(q_val, axis=-1)
        # ѵ��ģʽ��ʹ��̽���㷨
        elif a_type == "train":
            return self.selector.select_action(state)

        return None

    def optimize(self, experience):
        """
        ����������
        :param experience: ����Ԫ��
        :return:
        """
        index = None
        weights = None

        # �Ƿ�Ϊ���Ⱦ���ط�
        per = True if len(experience) == 7 else False
        # ���
        if per:
            index, weights, state, action, next_state, reward, done = experience
        else:
            state, action, next_state, reward, done = experience

        # ������ת��Ϊ��������
        action = action.squeeze().long()

        # ����td_target
        td_target = self.calculate_td_target(next_state, reward, done)
        # ����y
        q_sa = self.policy_model(state).gather(1, action)
        # ����td���
        td_error = td_target.detach() - q_sa
        # �����ֵ��ʧ
        if per:
            value_loss = (weights * td_error).pow(2).mul(0.5).mean()
        else:
            value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.value_optimizer.step()

        # ������1
        self.step += 1
        # �����������
        if self.step % self.update_interval == 0:
            self.update_network()

        # ���ΪPER���򷵻����������ȼ�
        if per:
            return index, td_error.squeeze().detach().cpu().numpy()

        return None

    def update_network(self) -> None:
        """
        �����������
        :return:
        """
        self.mix_network(self.target_model, self.policy_model, self.tau)

    def calculate_td_target(self, next_state, reward, done) -> torch.Tensor:
        """
        ����TDĿ��
        :return:
        """
        ...


class PGBasedAgent(Agent):
    """
    ���ڲ����ݶȵ������壬REINFORCE, VPG, A3C, A2C, PPO
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 value_model: net.ValueModel | None = None,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 grad_norm_clipping: float = 1.0,
                 gae: bool = False,
                 gae_lambda: float | None = 0.95,
                 entropy_loss_weight: float | None = 0.01):
        """
        ���캯��
        """
        super(PGBasedAgent, self).__init__(gamma, lr, None, None, grad_norm_clipping)

        # �Ƿ�ʹ��GAE
        self.gae = gae
        # GAE����Ȩ��
        self.gae_lambda = gae_lambda
        # ����ʧȨ��
        self.entropy_loss_weight = entropy_loss_weight

        # ��������
        self.policy_model = policy_model
        # ��ֵ����
        self.value_model = value_model
        # ������ʼ��
        self.policy_model.apply(self.policy_model.orthogonal_init)
        if self.value_model is not None:
            self.value_model.apply(self.value_model.orthogonal_init)

        # �Ż���
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        if self.value_model is not None:
            self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.lr)

        # �����豸
        self.device: torch.device = self.policy_model.device

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        ѡ����
        :param state: ��ǰ״̬
        :param a_type: ģʽ, train, eval
        :return:
        """
        # ״̬ת��Ϊ����
        state = torch.tensor(state, dtype=torch.float, device=self.device).requires_grad_(True)

        # ��Ҫ�ݶȴ���
        action, log_prob, entropy, best_action = self.policy_model.full_pass(state)

        if a_type == "eval":
            return best_action.detach().cpu().numpy() if best_action.numel() > 1 else best_action.item()
        elif a_type == "train":
            return action.detach().cpu().numpy() if action.numel() > 1 else action.item(), log_prob, entropy

        return None


class ACBasedAgent(Agent):
    """
    ����Actor-Critic�ܹ���������: DDPG, TD3, SAC
    """

    def __init__(self,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0):
        """
        ���캯��
        """
        super(ACBasedAgent, self).__init__(gamma, lr, tau, update_interval, grad_norm_clipping)

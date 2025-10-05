# -*- coding:gbk -*-

"""
强化学习单智能体
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
    单智能体基类
    """

    def __init__(self,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 tau: float | None = 0.1,
                 update_interval: int | None = 5,
                 grad_norm_clipping: float = 1.0
                 ):
        """
        构造函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param tau: 网络更新率
        :param grad_norm_clipping: 梯度裁剪最大值
        """
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_interval = update_interval
        self.grad_norm_clipping = grad_norm_clipping

        # 当前步数
        self.step = 0
        # 动作选择器
        self.selector = None
        # 策略网络
        self.policy_model = None
        # 张量设备
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        选择动作
        :param state: 当前状态
        :param a_type: 动作选择类型, train, eval
        :return:
        """
        ...

    def optimize(self, *args, **kwargs):
        """
        智能体更新
        :return:
        """
        ...

    def update_network(self) -> None:
        """
        更新目标网络
        :return:
        """
        ...

    def save_policy(self, filename) -> None:
        """
        保存策略
        :param filename: 策略路径
        :return:
        """
        torch.save(self.policy_model.state_dict(), filename)

    def load_policy(self, filename) -> None:
        """
        加载策略
        :param filename: 策略路径
        :return:
        """
        self.policy_model.load_state_dict(torch.load(filename, weights_only=True))

    @staticmethod
    def mix_network(target_network: torch.nn.Module, online_network: torch.nn.Module, tau: float):
        """
        混合网络权重
        :param target_network: 目标网络
        :param online_network: 在线网络
        :param tau: 混合权重
        :return:
        """
        for target, online in zip(target_network.parameters(), online_network.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)


class ValueBasedAgent(Agent):
    """
    基于价值的智能体：DQN, Double DQN及变体
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
        构造函数
        """
        super(ValueBasedAgent, self).__init__(gamma, lr, tau, update_interval, grad_norm_clipping)

        # 动作选择器
        self.selector = selector

        # 目标网络和在线网络
        self.policy_model = policy_model
        self.target_model = target_model
        # 同步权重
        self.target_model.load_state_dict(self.policy_model.state_dict())
        # 优化器
        self.value_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)

        # 张量设备
        self.device: torch.device = self.policy_model.device

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        选择动作
        :param state: 当前状态
        :param a_type: 动作选择类型, train, eval
        :return:
        """
        # 状态转换为张量
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # 评估模式选择动作，采用贪婪算法
        if a_type == "eval":
            with torch.no_grad():
                q_val = self.policy_model(state).detach().squeeze().cpu().numpy()
            return np.argmax(q_val, axis=-1)
        # 训练模式，使用探索算法
        elif a_type == "train":
            return self.selector.select_action(state)

        return None

    def optimize(self, experience):
        """
        更新智能体
        :param experience: 经验元组
        :return:
        """
        index = None
        weights = None

        # 是否为优先经验回放
        per = True if len(experience) == 7 else False
        # 解包
        if per:
            index, weights, state, action, next_state, reward, done = experience
        else:
            state, action, next_state, reward, done = experience

        # 将动作转换为索引类型
        action = action.squeeze().long()

        # 计算td_target
        td_target = self.calculate_td_target(next_state, reward, done)
        # 计算y
        q_sa = self.policy_model(state).gather(1, action)
        # 计算td误差
        td_error = td_target.detach() - q_sa
        # 计算价值损失
        if per:
            value_loss = (weights * td_error).pow(2).mul(0.5).mean()
        else:
            value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.value_optimizer.step()

        # 步数加1
        self.step += 1
        # 更新网络参数
        if self.step % self.update_interval == 0:
            self.update_network()

        # 如果为PER，则返回索引和优先级
        if per:
            return index, td_error.squeeze().detach().cpu().numpy()

        return None

    def update_network(self) -> None:
        """
        更新网络参数
        :return:
        """
        self.mix_network(self.target_model, self.policy_model, self.tau)

    def calculate_td_target(self, next_state, reward, done) -> torch.Tensor:
        """
        计算TD目标
        :return:
        """
        ...


class PGBasedAgent(Agent):
    """
    基于策略梯度的智能体，REINFORCE, VPG, A3C, A2C, PPO
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
        构造函数
        """
        super(PGBasedAgent, self).__init__(gamma, lr, None, None, grad_norm_clipping)

        # 是否使用GAE
        self.gae = gae
        # GAE估计权重
        self.gae_lambda = gae_lambda
        # 熵损失权重
        self.entropy_loss_weight = entropy_loss_weight

        # 策略网络
        self.policy_model = policy_model
        # 价值网络
        self.value_model = value_model
        # 正交初始化
        self.policy_model.apply(self.policy_model.orthogonal_init)
        if self.value_model is not None:
            self.value_model.apply(self.value_model.orthogonal_init)

        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        if self.value_model is not None:
            self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.lr)

        # 张量设备
        self.device: torch.device = self.policy_model.device

    def select_action(self, state: np.ndarray, a_type: str = "train"):
        """
        选择动作
        :param state: 当前状态
        :param a_type: 模式, train, eval
        :return:
        """
        # 状态转换为张量
        state = torch.tensor(state, dtype=torch.float, device=self.device).requires_grad_(True)

        # 需要梯度传播
        action, log_prob, entropy, best_action = self.policy_model.full_pass(state)

        if a_type == "eval":
            return best_action.detach().cpu().numpy() if best_action.numel() > 1 else best_action.item()
        elif a_type == "train":
            return action.detach().cpu().numpy() if action.numel() > 1 else action.item(), log_prob, entropy

        return None


class ACBasedAgent(Agent):
    """
    基于Actor-Critic架构的智能体: DDPG, TD3, SAC
    """

    def __init__(self,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0):
        """
        构造函数
        """
        super(ACBasedAgent, self).__init__(gamma, lr, tau, update_interval, grad_norm_clipping)

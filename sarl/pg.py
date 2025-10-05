# -*- coding:gbk -*-

"""
基于策略梯度的智能体
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
    REINFORCE智能体
    """

    def __init__(self,
                 policy_model: net.PolicyModel,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 grad_norm_clipping: float = 1.0,
                 ):
        """
        构造函数
        :param policy_model: 策略网络
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        """
        super(REINFORCEAgent, self).__init__(policy_model, None, gamma, lr, grad_norm_clipping,
                                             False, None, None)

    def optimize(self, trajectory):
        """
        优化智能体
        :param trajectory: 智能体轨迹
        :return:
        """
        # 解包
        states, actions, returns = trajectory
        # 动作评估
        log_pa, _ = self.policy_model.evaluate_actions(states, actions)

        # 优势函数
        advantage = returns
        # 计算策略损失
        policy_loss = -(advantage.detach() * log_pa).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.policy_optimizer.step()


class VPGAgent(agent.PGBasedAgent):
    """
    VPG智能体
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
        构造函数
        :param policy_model: 策略网络
        :param value_model: 价值网络
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param entropy_loss_weight: 熵损失权重
        """
        super(VPGAgent, self).__init__(policy_model, value_model, gamma, lr, grad_norm_clipping, False, None,
                                       entropy_loss_weight)

    def optimize(self, trajectory: tuple):
        """
        优化智能体
        :param trajectory: 智能体轨迹
        :return:
        """
        # 解包
        states, actions, returns = trajectory
        # 动作评估
        log_pa, entropy = self.policy_model.evaluate_actions(states, actions)
        # 状态值
        values = self.value_model(states)

        # 计算优势函数
        advantage = returns.detach() - values
        # 计算策略损失
        policy_loss = -(advantage.detach() * log_pa).mean() - self.entropy_loss_weight * entropy.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        self.policy_optimizer.step()

        # 计算价值损失
        value_error = advantage
        value_loss = value_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
        self.value_optimizer.step()


class A3CAgent(agent.PGBasedAgent):
    """
    A3C智能体
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
        构造函数
        :param shared_policy_model: 共享策略网络
        :param shared_value_model: 共享价值网络
        :param shared_policy_optimizer: 共享策略网络优化器
        :param shared_value_optimizer: 共享价值网络优化器
        :param policy_model: 本地策略网络
        :param value_model: 本地价值网络
        :param gamma: 折扣系数
        :param grad_norm_clipping:  梯度裁剪范围
        :param entropy_loss_weight: 熵损失权重
        """
        super(A3CAgent, self).__init__(policy_model, value_model, gamma, 0, grad_norm_clipping, False, None,
                                       entropy_loss_weight)

        # 共享模型
        self.shared_policy_model = shared_policy_model
        self.shared_value_model = shared_value_model
        # 共享优化器
        self.policy_optimizer = shared_policy_optimizer
        self.value_optimizer = shared_value_optimizer

        # 参数同步
        self.policy_model.load_state_dict(self.shared_policy_model.state_dict())
        self.value_model.load_state_dict(self.shared_value_model.state_dict())

    def optimize(self, trajectory):
        """
        优化智能体
        :param trajectory: 智能体轨迹
        :return:
        """
        # 解包
        states, actions, returns = trajectory
        # 动作评估
        log_pa, entropy = self.policy_model.evaluate_actions(states, actions)
        # 状态值
        values = self.value_model(states)

        # 计算优势函数
        advantage = returns.detach() - values
        # 计算策略损失
        policy_loss = -(advantage.detach() * log_pa).mean() - self.entropy_loss_weight * entropy.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
        # 本地模型的梯度复制到共享模型
        for param, shared_param in zip(self.policy_model.parameters(), self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.policy_optimizer.step()
        # 同步参数
        self.policy_model.load_state_dict(self.shared_policy_model.state_dict())

        # 计算价值损失
        value_error = advantage
        value_loss = value_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
        # 本地模型的梯度复制到共享模型
        for param, shared_param in zip(self.value_model.parameters(), self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.value_optimizer.step()
        # 同步参数
        self.value_model.load_state_dict(self.shared_value_model.state_dict())


class A2CAgent(agent.PGBasedAgent):
    """
    A2C智能体
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
        构造函数
        :param policy_model: 策略网络
        :param value_model: 价值网络
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param gae_lambda: GAE的奖励折扣系数
        :param entropy_loss_weight: 熵损失权重
        :param gae_norm: 是否标准化GAE
        """
        super(A2CAgent, self).__init__(policy_model, value_model, gamma, lr, grad_norm_clipping, True, gae_lambda,
                                       entropy_loss_weight)

        # 是否标准化GAE
        self.gae_norm = gae_norm

    def optimize(self,
                 buffer: utils.buffer.RolloutBuffer,
                 seed=None):
        """
        优化模型
        :return:
        """
        # 轨迹模拟
        buffer.roll_out(self, seed)

        # 一次性输出所有数据，只循环一次
        for trajectory in buffer.sample(total=True):
            # 解包
            states, actions, _, _, advantages, returns = trajectory
            # 对当前策略进行评估
            log_prob, entropy = self.policy_model.evaluate_actions(states, actions)
            # 计算价状态值
            values = self.value_model(states)

            # 标准化GAE
            if self.gae_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # 策略损失
            policy_loss = -(advantages.detach() * log_prob).mean() - self.entropy_loss_weight * entropy.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
            self.policy_optimizer.step()

            # 价值损失
            value_loss = (returns.detach() - values).pow(2).mul(0.5).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
            self.value_optimizer.step()

        # 清除缓冲区
        buffer.clear()


class PPOAgent(A2CAgent):
    """
    PPO智能体
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
        构造函数
        :param policy_model: 策略网络
        :param value_model: 价值网络
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param gae_lambda: GAE的奖励折扣系数
        :param entropy_loss_weight: 熵损失权重
        :param gae_norm: 是否标准化GAE
        :param update_epoch: 网络重复优化次数
        :param policy_clipping: 策略新旧概率比裁剪范围
        :param value_clipping: 新旧价值之差裁剪范围
        """
        super(PPOAgent, self).__init__(policy_model, value_model, gamma, lr, grad_norm_clipping, gae_lambda,
                                       entropy_loss_weight, gae_norm)

        assert 0 <= policy_clipping, f"策略裁剪范围因子应大于等于0，实际为 {policy_clipping}"
        assert 0 <= value_clipping, f"价值裁剪范围因子应大于等于0，实际为 {value_clipping}"

        self.update_epoch = update_epoch
        self.policy_clipping = policy_clipping
        self.value_clipping = value_clipping

    def optimize(self,
                 buffer: utils.buffer.RolloutBuffer,
                 seed=None
                 ):
        """
        优化智能体
        :param buffer: 轨迹生成抽样缓冲区
        :param seed: 环境随机种子
        :return:
        """
        # 模拟轨迹
        buffer.roll_out(self, seed)

        # 网络更新次数
        for _ in range(self.update_epoch):
            # 将数据切分成mini-batch，分批次更新网络
            for trajectory in buffer.sample(total=False):
                # 采样
                states, actions, log_prob, values, advantages, returns = trajectory

                # 对GAE标准化
                if self.gae_norm:
                    advantages = (advantages - torch.mean(advantages).item()) / (torch.std(advantages).item() + 1e-6)

                # 新策略下的动作对数概率和熵
                new_log_prob, new_entropy = self.policy_model.evaluate_actions(states, actions)
                # 新价值网络下的状态值
                new_values = self.value_model(states)

                # 新旧策略概率比
                ratio = torch.exp(new_log_prob - log_prob.detach())
                # 原始优化目标
                pi_obj = ratio * advantages.detach()
                # 裁剪后的优化目标
                pi_obj_clipped = torch.clip(ratio, 1.0 - self.policy_clipping,
                                            1.0 + self.policy_clipping) * advantages.detach()
                # 策略损失函数
                policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean() - self.entropy_loss_weight * new_entropy.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_norm_clipping)
                self.policy_optimizer.step()

                # 对新的状态值进行裁剪
                new_value_clipped = values.detach() + torch.clip((new_values - values.detach()),
                                                                 -self.value_clipping, self.value_clipping)
                # 原始的价值损失
                v_obj = (returns.detach() - new_values).pow(2)
                # 裁剪后的价值损失
                v_obj_clipped = (returns.detach() - new_value_clipped).pow(2)
                # 价值损失
                value_loss = torch.max(v_obj, v_obj_clipped).mul(0.5).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.grad_norm_clipping)
                self.value_optimizer.step()

        # 清除轨迹生成缓冲区的数据
        buffer.clear()

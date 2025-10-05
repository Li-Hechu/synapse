# -*- coding:gbk -*-

"""
动作探索类型
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = ["Selector",
           "LinearSelector",
           "SoftmaxSelector",
           "UCBSelector",
           "ExponentialSelector",
           "ProbabilitySelector",
           "GaussianSelector"]


class Selector(object):
    """
    动作选择器
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        构造函数
        :param max_val: 最大值
        :param min_val: 最小值
        :param decay_step: 衰减步长
        :param model: 策略模型
        """
        self.max_val = max_val
        self.min_val = min_val
        self.decay_step = decay_step
        self.model = model
        # 类型
        self.dtype = None
        # epsilon
        self.epsilon = None
        # 当前步数
        self.step = 0
        # 是否首次访问
        self.first_visited = True
        # 动作维度
        self.action_dim = 0

        assert self.min_val <= self.max_val, "最大值小于最小值"
        assert self.decay_step >= 0, "衰减步数应该大于0"

    def update(self):
        """
        更新当前步数
        :return:
        """
        self.step += 1

    def select_action(self, state):
        """
        选择动作
        :param state: 当前状态
        :return:
        """
        # 当前epsilon值
        epsilon = self.epsilon[self.step] if self.step < self.decay_step else self.min_val

        # 首次访问
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().cpu().numpy()
            # 获取动作维度
            self.action_dim = action.shape[-1]
            # 改变状态
            self.first_visited = False

        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_val = self.model(state).detach().squeeze().cpu().numpy()
            action = np.argmax(q_val, axis=-1)

        # 更新
        self.update()

        return action

    def draw(self):
        """
        绘制epsilon曲线
        :return:
        """
        plt.figure(1)
        plt.plot(self.epsilon)
        plt.show()


class LinearSelector(Selector):
    """
    线性衰减动作选择器
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        构造函数
        :param max_val: 最大值
        :param min_val: 最小值
        :param decay_step: 衰减步长a衰减类型m dtype:
        :param model: 策略模型
        """
        super(LinearSelector, self).__init__(max_val, min_val, decay_step, model)

        assert 0 <= self.min_val <= self.max_val <= 1, "衰减参数最大最小值应该位于[0,1]区间内"

        self.dtype = "linear"
        self.epsilon = np.linspace(self.max_val, self.min_val, self.decay_step)


class ExponentialSelector(Selector):
    """
    指数衰减动作选择器
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        构造函数
        :param max_val: 最大值
        :param min_val: 最小值
        :param decay_step: 衰减步长
        :param model: 策略模型
        """
        super(ExponentialSelector, self).__init__(max_val, min_val, decay_step, model)

        assert 0 <= self.min_val <= self.max_val <= 1, "衰减参数最大最小值应该位于[0,1]区间内"

        self.dtype = "exp"
        self.epsilon = 0.01 / np.logspace(-2, 0, self.decay_step, endpoint=False) - 0.01
        self.epsilons = self.epsilon * (self.max_val - self.min_val) + self.min_val


class SoftmaxSelector(Selector):
    """
    softmax动作选择器
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        构造函数
        :param max_val: 最大值
        :param min_val: 最小值
        :param decay_step: 衰减步长
        :param model: 策略模型
        """
        super(SoftmaxSelector, self).__init__(max_val, min_val, decay_step, model)

        self.dtype = "softmax"

    def update(self):
        """
        更新参数
        :return:
        """
        # 温度参数
        temperature = np.clip(1 - self.step / self.decay_step, 0, 1)
        temperature = (self.max_val - self.min_val) * temperature + self.min_val
        self.step += 1

        return temperature

    def select_action(self, state):
        """
        选择动作
        :param state: 当前状态
        :return:
        """
        # 获取温度参数
        temperature = self.update()

        # 首次访问
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().cpu().numpy()
            # 获取动作维度
            self.action_dim = action.shape[-1]
            # 改变状态
            self.first_visited = False

        with torch.no_grad():
            q_val = self.model(state).detach().squeeze().cpu().numpy()

        scaled_qs = q_val / temperature
        norm_qs = scaled_qs - scaled_qs.max()
        e = np.exp(norm_qs)
        probs = e / np.sum(e)

        action = np.random.choice(np.arange(self.action_dim), size=1, p=probs)

        return action


class UCBSelector(Selector):
    """
    置信上界动作选择器
    """
    def __init__(self, c: float, model: torch.nn.Module):
        """
        构造函数
        :param c: 探索比率
        :param model: 策略模型
        """
        super(UCBSelector, self).__init__(0, 0, 0, model)

        self.c = c
        self.dtype = "ucb"
        # 动作计数器
        self.counter = None

    def select_action(self, state):
        """
        选择动作
        :param state: 当前状态
        :return:
        """
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().squeeze().cpu().numpy()
            # 获取动作维度
            self.action_dim = action.shape[-1]
            # 初始化计数器
            self.counter = np.zeros(self.action_dim, dtype=np.float32)
            # 改变状态
            self.first_visited = False

        with torch.no_grad():
            q_val = self.model(state).detach().squeeze().cpu().numpy()

        action = np.argmax(q_val + self.c * np.sqrt(np.log(self.step + 1e-6) / (self.counter + 1e-6)), axis=-1)

        # 更新计数器
        self.counter[action] += 1
        # 更新步数
        self.update()

        return action


class ProbabilitySelector(Selector):
    """
    概率动作选择器
    """
    def __init__(self, model: torch.nn.Module):
        """
        构造函数
        """
        super(ProbabilitySelector, self).__init__(0, 0, 0, model)

    def select_action(self, state):
        """
        选择动作
        :param state: 当前状态
        :return:
        """
        logits = self.model(state)
        # 获取分布
        dist = torch.distributions.Categorical(logits=logits)
        # 采样动作
        action = dist.sample()

        return action.detach().cpu().numpy()


class GaussianSelector(Selector):
    """
    高斯噪声动作选择器
    """
    def __init__(self, max_val, min_val, decay_step, model: torch.nn.Module, action_bounds: tuple):
        """
        构造函数
        :param max_val: 标准差最大值
        :param min_val: 标准差最小值
        :param decay_step: 标准差衰减步数
        :param action_bounds: 动作区间
        :param model: 策略模型
        """
        super(GaussianSelector, self).__init__(max_val, min_val, decay_step, model)

        assert self.min_val >= 0, "标准差衰减最小值应大于等于0"

        self.low, self.high = action_bounds
        self.dtype = "gaussian"
        # 标准差衰减
        self.epsilon = np.linspace(self.max_val, self.min_val, self.decay_step)

    def select_action(self, state):
        """
        动作选择
        :param state: 当前状态
        :return:
        """
        # 获取标准差
        sigma = self.epsilon[self.step] if self.step < self.decay_step else self.min_val

        # 首次访问
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().cpu().numpy()
            # 获取动作维度
            self.action_dim = action.shape[-1]
            # 改变状态
            self.first_visited = False

        # 获取动作
        with torch.no_grad():
            action = self.model(state).detach().squeeze().cpu().numpy()

        # 生成噪声
        noise = np.random.normal(loc=0, scale=sigma, size=(self.action_dim,))
        # 叠加
        action = np.clip(action + noise, self.low, self.high)

        # 更新步数
        self.update()

        return action

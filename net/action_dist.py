# -*- coding:gbk -*-


import torch

__all__ = ["BaseDistribution", "NormalDistribution", "CategoricalDistribution", "DeterministicDistribution"]


class BaseDistribution(object):
    """
    基本动作分布
    """
    def __init__(self, action_dim: int, device: str):
        """
        构造函数
        :param action_dim: 动作维度
        """
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.distribution = None

    def update_distribution(self, *arg, **kwargs):
        """
        更新概率分布
        :param arg:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def log_prob(self, action: torch.Tensor):
        """
        获取给定动作对应的对数概率
        :param action: 动作
        :return:
        """
        raise NotImplementedError

    def entropy(self):
        """
        获取分布熵
        :return:
        """
        raise NotImplementedError

    def sample(self):
        """
        从分布中获取动作及其概率
        :return:
        """
        raise NotImplementedError

    def mode(self):
        """
        获取最佳动作
        :return:
        """
        return self.distribution.mode

    def get_action(self, deterministic=False):
        """
        是否选择确定性动作
        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()



class CategoricalDistribution(BaseDistribution):
    """
    离散动作概率分布
    """
    def __init__(self, action_dim: int, device: str):
        """
        构造函数
        :param action_dim:
        """
        super(CategoricalDistribution, self).__init__(action_dim, device)

    def update_distribution(self, logits: torch.Tensor):
        """
        更新动作概率分布
        :param logits:
        :return:
        """
        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action: torch.Tensor):
        return self.distribution.log_prob(action).unsqueeze(-1)

    def entropy(self):
        return self.distribution.entropy().unsqueeze(-1)

    def sample(self):
        action = self.distribution.sample()
        log_prob = self.distribution.log_prob(action)
        return action, log_prob.unsqueeze(-1)


class NormalDistribution(BaseDistribution):
    """
    高斯动作概率分布
    """
    def __init__(self, action_dim: int, bounds: tuple, device: str):
        """
        构造函数
        :param action_dim:
        :param device:
        """
        super(NormalDistribution, self).__init__(action_dim, device)

        # 连续动作区间
        self.action_max = torch.tensor(bounds[1], device=device, dtype=torch.float)
        self.action_min = torch.tensor(bounds[0], device=device, dtype=torch.float)
        # 缩放动作区间
        self.nn_min = torch.tensor([-1], dtype=torch.float, device=self.device)
        self.nn_max = torch.tensor([1], dtype=torch.float, device=self.device)

    def update_distribution(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        更新分布
        :param mean:
        :param log_std:
        :return:
        """
        self.distribution = torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, action: torch.Tensor):
        # 将动作反缩放，得到压缩后的动作
        compressed_action = self._inv_scale(action)
        # 反压缩得到原始动作
        pre_action = torch.atanh(compressed_action)
        # 求解对数概率
        log_prob = self.distribution.log_prob(pre_action) - torch.log((1 - compressed_action.pow(2)).clamp(0, 1) + 1e-6)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        return log_prob

    def entropy(self):
        return torch.sum(self.distribution.entropy(), dim=-1, keepdim=True)

    def sample(self):
        # 采样动作，可微分采样
        pre_action = self.distribution.rsample()
        # 压缩到[-1,1]区间
        compressed_action = torch.tanh(pre_action)
        # 将压缩动作放缩到给定区间内
        action = self._scale(compressed_action)
        # 对数概率，第二项是对tanh压缩操作的概率密度修正
        log_prob = self.distribution.log_prob(pre_action) - torch.log((1 - compressed_action.pow(2)).clamp(0, 1) + 1e-6)
        # 对所有动作的log_prob求和
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        return action, log_prob

    def _scale(self, x: torch.Tensor):
        """
        连续动作放缩
        :param x:
        :return:
        """
        return (x - self.nn_min) * (self.action_max - self.action_min) / (self.nn_max - self.nn_min) + self.action_min

    def _inv_scale(self, x: torch.Tensor):
        """
        连续动作反放缩
        :param x:
        :return:
        """
        return (x - self.action_min) * (self.nn_max - self.nn_min) / (self.action_max - self.action_min) + self.nn_min


class DeterministicDistribution(BaseDistribution):
    """
    确定性分布
    """
    def __init__(self, action_dim: int, bounds: tuple, device: str):
        """
        构造函数
        :param action_dim:
        :param bounds:
        :param device:
        """
        super(DeterministicDistribution, self).__init__(action_dim, device)

        # 连续动作区间
        self.action_max = torch.tensor(bounds[1], device=device, dtype=torch.float)
        self.action_min = torch.tensor(bounds[0], device=device, dtype=torch.float)

    def update_distribution(self, action: torch.Tensor):
        """
        更新分布
        :param action:
        :return:
        """
        # 动作范围裁剪
        self.distribution = torch.clip(action, self.action_min, self.action_max)

    def log_prob(self, action: torch.Tensor):
        # 对数概率维度
        log_prob_dim = (*action.shape[:-1], 1)
        # 对数概率
        log_prob = torch.zeros(log_prob_dim, dtype=torch.float, device=self.device)
        return log_prob

    def entropy(self):
        entropy_dim = (*self.distribution.shape[:-1], 1)
        entropy = torch.zeros(entropy_dim, dtype=torch.float, device=self.device)
        return entropy

    def sample(self):
        return self.distribution, self.log_prob(self.distribution)

# -*- coding:gbk -*-


import torch

__all__ = ["BaseDistribution", "NormalDistribution", "CategoricalDistribution", "DeterministicDistribution"]


class BaseDistribution(object):
    """
    ���������ֲ�
    """
    def __init__(self, action_dim: int, device: str):
        """
        ���캯��
        :param action_dim: ����ά��
        """
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.distribution = None

    def update_distribution(self, *arg, **kwargs):
        """
        ���¸��ʷֲ�
        :param arg:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def log_prob(self, action: torch.Tensor):
        """
        ��ȡ����������Ӧ�Ķ�������
        :param action: ����
        :return:
        """
        raise NotImplementedError

    def entropy(self):
        """
        ��ȡ�ֲ���
        :return:
        """
        raise NotImplementedError

    def sample(self):
        """
        �ӷֲ��л�ȡ�����������
        :return:
        """
        raise NotImplementedError

    def mode(self):
        """
        ��ȡ��Ѷ���
        :return:
        """
        return self.distribution.mode

    def get_action(self, deterministic=False):
        """
        �Ƿ�ѡ��ȷ���Զ���
        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()



class CategoricalDistribution(BaseDistribution):
    """
    ��ɢ�������ʷֲ�
    """
    def __init__(self, action_dim: int, device: str):
        """
        ���캯��
        :param action_dim:
        """
        super(CategoricalDistribution, self).__init__(action_dim, device)

    def update_distribution(self, logits: torch.Tensor):
        """
        ���¶������ʷֲ�
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
    ��˹�������ʷֲ�
    """
    def __init__(self, action_dim: int, bounds: tuple, device: str):
        """
        ���캯��
        :param action_dim:
        :param device:
        """
        super(NormalDistribution, self).__init__(action_dim, device)

        # ������������
        self.action_max = torch.tensor(bounds[1], device=device, dtype=torch.float)
        self.action_min = torch.tensor(bounds[0], device=device, dtype=torch.float)
        # ���Ŷ�������
        self.nn_min = torch.tensor([-1], dtype=torch.float, device=self.device)
        self.nn_max = torch.tensor([1], dtype=torch.float, device=self.device)

    def update_distribution(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        ���·ֲ�
        :param mean:
        :param log_std:
        :return:
        """
        self.distribution = torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, action: torch.Tensor):
        # �����������ţ��õ�ѹ����Ķ���
        compressed_action = self._inv_scale(action)
        # ��ѹ���õ�ԭʼ����
        pre_action = torch.atanh(compressed_action)
        # ����������
        log_prob = self.distribution.log_prob(pre_action) - torch.log((1 - compressed_action.pow(2)).clamp(0, 1) + 1e-6)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        return log_prob

    def entropy(self):
        return torch.sum(self.distribution.entropy(), dim=-1, keepdim=True)

    def sample(self):
        # ������������΢�ֲ���
        pre_action = self.distribution.rsample()
        # ѹ����[-1,1]����
        compressed_action = torch.tanh(pre_action)
        # ��ѹ����������������������
        action = self._scale(compressed_action)
        # �������ʣ��ڶ����Ƕ�tanhѹ�������ĸ����ܶ�����
        log_prob = self.distribution.log_prob(pre_action) - torch.log((1 - compressed_action.pow(2)).clamp(0, 1) + 1e-6)
        # �����ж�����log_prob���
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        return action, log_prob

    def _scale(self, x: torch.Tensor):
        """
        ������������
        :param x:
        :return:
        """
        return (x - self.nn_min) * (self.action_max - self.action_min) / (self.nn_max - self.nn_min) + self.action_min

    def _inv_scale(self, x: torch.Tensor):
        """
        ��������������
        :param x:
        :return:
        """
        return (x - self.action_min) * (self.nn_max - self.nn_min) / (self.action_max - self.action_min) + self.nn_min


class DeterministicDistribution(BaseDistribution):
    """
    ȷ���Էֲ�
    """
    def __init__(self, action_dim: int, bounds: tuple, device: str):
        """
        ���캯��
        :param action_dim:
        :param bounds:
        :param device:
        """
        super(DeterministicDistribution, self).__init__(action_dim, device)

        # ������������
        self.action_max = torch.tensor(bounds[1], device=device, dtype=torch.float)
        self.action_min = torch.tensor(bounds[0], device=device, dtype=torch.float)

    def update_distribution(self, action: torch.Tensor):
        """
        ���·ֲ�
        :param action:
        :return:
        """
        # ������Χ�ü�
        self.distribution = torch.clip(action, self.action_min, self.action_max)

    def log_prob(self, action: torch.Tensor):
        # ��������ά��
        log_prob_dim = (*action.shape[:-1], 1)
        # ��������
        log_prob = torch.zeros(log_prob_dim, dtype=torch.float, device=self.device)
        return log_prob

    def entropy(self):
        entropy_dim = (*self.distribution.shape[:-1], 1)
        entropy = torch.zeros(entropy_dim, dtype=torch.float, device=self.device)
        return entropy

    def sample(self):
        return self.distribution, self.log_prob(self.distribution)

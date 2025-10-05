# -*- coding:gbk -*-

"""
����̽������
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
    ����ѡ����
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        ���캯��
        :param max_val: ���ֵ
        :param min_val: ��Сֵ
        :param decay_step: ˥������
        :param model: ����ģ��
        """
        self.max_val = max_val
        self.min_val = min_val
        self.decay_step = decay_step
        self.model = model
        # ����
        self.dtype = None
        # epsilon
        self.epsilon = None
        # ��ǰ����
        self.step = 0
        # �Ƿ��״η���
        self.first_visited = True
        # ����ά��
        self.action_dim = 0

        assert self.min_val <= self.max_val, "���ֵС����Сֵ"
        assert self.decay_step >= 0, "˥������Ӧ�ô���0"

    def update(self):
        """
        ���µ�ǰ����
        :return:
        """
        self.step += 1

    def select_action(self, state):
        """
        ѡ����
        :param state: ��ǰ״̬
        :return:
        """
        # ��ǰepsilonֵ
        epsilon = self.epsilon[self.step] if self.step < self.decay_step else self.min_val

        # �״η���
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().cpu().numpy()
            # ��ȡ����ά��
            self.action_dim = action.shape[-1]
            # �ı�״̬
            self.first_visited = False

        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_val = self.model(state).detach().squeeze().cpu().numpy()
            action = np.argmax(q_val, axis=-1)

        # ����
        self.update()

        return action

    def draw(self):
        """
        ����epsilon����
        :return:
        """
        plt.figure(1)
        plt.plot(self.epsilon)
        plt.show()


class LinearSelector(Selector):
    """
    ����˥������ѡ����
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        ���캯��
        :param max_val: ���ֵ
        :param min_val: ��Сֵ
        :param decay_step: ˥������a˥������m dtype:
        :param model: ����ģ��
        """
        super(LinearSelector, self).__init__(max_val, min_val, decay_step, model)

        assert 0 <= self.min_val <= self.max_val <= 1, "˥�����������СֵӦ��λ��[0,1]������"

        self.dtype = "linear"
        self.epsilon = np.linspace(self.max_val, self.min_val, self.decay_step)


class ExponentialSelector(Selector):
    """
    ָ��˥������ѡ����
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        ���캯��
        :param max_val: ���ֵ
        :param min_val: ��Сֵ
        :param decay_step: ˥������
        :param model: ����ģ��
        """
        super(ExponentialSelector, self).__init__(max_val, min_val, decay_step, model)

        assert 0 <= self.min_val <= self.max_val <= 1, "˥�����������СֵӦ��λ��[0,1]������"

        self.dtype = "exp"
        self.epsilon = 0.01 / np.logspace(-2, 0, self.decay_step, endpoint=False) - 0.01
        self.epsilons = self.epsilon * (self.max_val - self.min_val) + self.min_val


class SoftmaxSelector(Selector):
    """
    softmax����ѡ����
    """
    def __init__(self, max_val: float, min_val: float, decay_step: int, model: torch.nn.Module):
        """
        ���캯��
        :param max_val: ���ֵ
        :param min_val: ��Сֵ
        :param decay_step: ˥������
        :param model: ����ģ��
        """
        super(SoftmaxSelector, self).__init__(max_val, min_val, decay_step, model)

        self.dtype = "softmax"

    def update(self):
        """
        ���²���
        :return:
        """
        # �¶Ȳ���
        temperature = np.clip(1 - self.step / self.decay_step, 0, 1)
        temperature = (self.max_val - self.min_val) * temperature + self.min_val
        self.step += 1

        return temperature

    def select_action(self, state):
        """
        ѡ����
        :param state: ��ǰ״̬
        :return:
        """
        # ��ȡ�¶Ȳ���
        temperature = self.update()

        # �״η���
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().cpu().numpy()
            # ��ȡ����ά��
            self.action_dim = action.shape[-1]
            # �ı�״̬
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
    �����Ͻ綯��ѡ����
    """
    def __init__(self, c: float, model: torch.nn.Module):
        """
        ���캯��
        :param c: ̽������
        :param model: ����ģ��
        """
        super(UCBSelector, self).__init__(0, 0, 0, model)

        self.c = c
        self.dtype = "ucb"
        # ����������
        self.counter = None

    def select_action(self, state):
        """
        ѡ����
        :param state: ��ǰ״̬
        :return:
        """
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().squeeze().cpu().numpy()
            # ��ȡ����ά��
            self.action_dim = action.shape[-1]
            # ��ʼ��������
            self.counter = np.zeros(self.action_dim, dtype=np.float32)
            # �ı�״̬
            self.first_visited = False

        with torch.no_grad():
            q_val = self.model(state).detach().squeeze().cpu().numpy()

        action = np.argmax(q_val + self.c * np.sqrt(np.log(self.step + 1e-6) / (self.counter + 1e-6)), axis=-1)

        # ���¼�����
        self.counter[action] += 1
        # ���²���
        self.update()

        return action


class ProbabilitySelector(Selector):
    """
    ���ʶ���ѡ����
    """
    def __init__(self, model: torch.nn.Module):
        """
        ���캯��
        """
        super(ProbabilitySelector, self).__init__(0, 0, 0, model)

    def select_action(self, state):
        """
        ѡ����
        :param state: ��ǰ״̬
        :return:
        """
        logits = self.model(state)
        # ��ȡ�ֲ�
        dist = torch.distributions.Categorical(logits=logits)
        # ��������
        action = dist.sample()

        return action.detach().cpu().numpy()


class GaussianSelector(Selector):
    """
    ��˹��������ѡ����
    """
    def __init__(self, max_val, min_val, decay_step, model: torch.nn.Module, action_bounds: tuple):
        """
        ���캯��
        :param max_val: ��׼�����ֵ
        :param min_val: ��׼����Сֵ
        :param decay_step: ��׼��˥������
        :param action_bounds: ��������
        :param model: ����ģ��
        """
        super(GaussianSelector, self).__init__(max_val, min_val, decay_step, model)

        assert self.min_val >= 0, "��׼��˥����СֵӦ���ڵ���0"

        self.low, self.high = action_bounds
        self.dtype = "gaussian"
        # ��׼��˥��
        self.epsilon = np.linspace(self.max_val, self.min_val, self.decay_step)

    def select_action(self, state):
        """
        ����ѡ��
        :param state: ��ǰ״̬
        :return:
        """
        # ��ȡ��׼��
        sigma = self.epsilon[self.step] if self.step < self.decay_step else self.min_val

        # �״η���
        if self.first_visited:
            with torch.no_grad():
                action = self.model(state).detach().cpu().numpy()
            # ��ȡ����ά��
            self.action_dim = action.shape[-1]
            # �ı�״̬
            self.first_visited = False

        # ��ȡ����
        with torch.no_grad():
            action = self.model(state).detach().squeeze().cpu().numpy()

        # ��������
        noise = np.random.normal(loc=0, scale=sigma, size=(self.action_dim,))
        # ����
        action = np.clip(action + noise, self.low, self.high)

        # ���²���
        self.update()

        return action

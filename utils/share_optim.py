# -*- coding:gbk -*-

"""
����̹����Ż���
"""

import torch

__all__ = ["SharedAdam",
           "SharedSGD",
           "SharedRMSprop",
           "SharedOptimizer"]


class SharedOptimizer:
    """
    �����Ż�������
    """

    def __init__(self, optimizer_cls, shared_state_spec, *args, **kwargs):
        """
        ���캯��
        :param optimizer_cls: �Ż�����
        :param shared_state_spec: ״̬�ֵ�
        :param args: �Ż�����������
        :param kwargs: �Ż�����������
        """
        # �����Ż���
        self.base_optimizer = optimizer_cls(*args, **kwargs)

        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                state = self.base_optimizer.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                # ���ݴ���Ĺ���״̬��񴴽�����
                for state_name, like_tensor in shared_state_spec.items():
                    state[state_name] = torch.zeros_like(
                        p.data if like_tensor == 'param' else torch.tensor(like_tensor)
                    ).share_memory_()

    def step(self, closure=None):
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.base_optimizer.state[p]['steps'] = self.base_optimizer.state[p]['shared_step'].item()
                self.base_optimizer.state[p]['shared_step'] += 1
        return self.base_optimizer.step(closure)

    def __getattr__(self, name):
        base = self.__dict__.get('base_optimizer')
        if base and hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class SharedAdam(SharedOptimizer):
    """
    ����Adam�Ż���
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        # ״̬�ֵ�
        shared_state_spec = {
            'exp_avg': 'param',
            'exp_avg_sq': 'param'
        }
        if weight_decay:
            shared_state_spec['weight_decay'] = 'param'
        if amsgrad:
            shared_state_spec['max_exp_avg_sq'] = 'param'
        super().__init__(torch.optim.Adam, shared_state_spec, params, lr=lr, betas=betas,
                         eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


class SharedRMSprop(SharedOptimizer):
    """
    ����RMSprop�Ż���
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        shared_state_spec = {
            'square_avg': 'param'
        }
        if weight_decay:
            shared_state_spec['weight_decay'] = 'param'
        if momentum > 0:
            shared_state_spec['momentum_buffer'] = 'param'
        if centered:
            shared_state_spec['grad_avg'] = 'param'
        super().__init__(torch.optim.RMSprop, shared_state_spec, params, lr=lr, alpha=alpha,
                         eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)


class SharedSGD(SharedOptimizer):
    """
    ����SGD�Ż���
    """
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        shared_state_spec = {}
        if momentum != 0:
            shared_state_spec['momentum_buffer'] = 'param'
        super().__init__(torch.optim.SGD, shared_state_spec, params, lr=lr,
                         momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov)

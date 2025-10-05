# -*- coding:gbk -*-

"""
多进程共享优化器
"""

import torch

__all__ = ["SharedAdam",
           "SharedSGD",
           "SharedRMSprop",
           "SharedOptimizer"]


class SharedOptimizer:
    """
    共享优化器基类
    """

    def __init__(self, optimizer_cls, shared_state_spec, *args, **kwargs):
        """
        构造函数
        :param optimizer_cls: 优化器类
        :param shared_state_spec: 状态字典
        :param args: 优化器创建参数
        :param kwargs: 优化器创建参数
        """
        # 创建优化器
        self.base_optimizer = optimizer_cls(*args, **kwargs)

        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                state = self.base_optimizer.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                # 根据传入的共享状态规格创建张量
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
    共享Adam优化器
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        # 状态字典
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
    共享RMSprop优化器
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
    共享SGD优化器
    """
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        shared_state_spec = {}
        if momentum != 0:
            shared_state_spec['momentum_buffer'] = 'param'
        super().__init__(torch.optim.SGD, shared_state_spec, params, lr=lr,
                         momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov)

# -*- coding:gbk -*-

"""
ѵ��������
"""

import net
import utils
import torch

__all__ = ["build_selector", "build_value_model", "build_policy_model", "build_replay_buffer"]


def build_selector(dtype: str,
                   model: torch.nn.Module,
                   max_val: float = None,
                   min_val: float = None,
                   decay_step: int = None,
                   c: float = None,
                   ):
    """
    ������ɢ����ѡ����
    :param dtype:
    :param model:
    :param max_val:
    :param min_val:
    :param decay_step:
    :param c:
    :return:
    """
    if dtype == 'linear':
        selector = utils.selector.LinearSelector(max_val, min_val, decay_step, model)
    elif dtype == 'exp':
        selector = utils.selector.ExponentialSelector(max_val, min_val, decay_step, model)
    elif dtype == 'softmax':
        selector = utils.selector.SoftmaxSelector(max_val, min_val, decay_step, model)
    elif dtype == 'ucb':
        selector = utils.selector.UCBSelector(c, model)
    else:
        assert f"δ֪ѡ�������ͣ�dtype ��Ϊ linear, exp, softmax, ucb"
        selector = None

    return selector


def build_replay_buffer(state_dim,
                        action_num,
                        buffer_size: int,
                        batch_size: int,
                        prioritized: bool = False,
                        alpha: float = None,
                        beta: float = None,
                        beta_inc_step: int = None,
                        device: str = "cuda" if torch.cuda.is_available() else "cpu", ):
    """
    �����طŻ�����
    :param state_dim:
    :param action_num:
    :param buffer_size:
    :param batch_size:
    :param prioritized:
    :param alpha:
    :param beta:
    :param beta_inc_step:
    :param device:
    :return:
    """
    if not prioritized:
        buffer = utils.buffer.SimpleReplayBuffer(state_dim, action_num, buffer_size, batch_size, device)
    else:
        assert alpha is not None and beta is not None, "�������ȻطŻ�������Ҫ��������alpha��beta"
        buffer = utils.buffer.PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta, beta_inc_step, device)

    return buffer


def build_policy_model(env_info: dict,
                       module_config: net.ModuleConfig,
                       deterministic: bool,
                       orth_init: bool,
                       device: str):
    """
    �����������
    :param env_info:
    :param module_config:
    :param deterministic:
    :param orth_init:
    :param device:
    :return:
    """
    # ������ȡ������
    extractor_kwargs = {
        "state_dim": env_info["state_dim"],
        "mlp_dim": module_config.policy_mlp_dim,
        "cnn_param": module_config.policy_cnn_param,
        "device": device
    }

    return net.PolicyModel(env_info["state_dim"],
                           env_info["action_dim"],
                           env_info["bounds"],
                           env_info["is_discrete"],
                           deterministic,
                           net.PolicyExtractor,
                           extractor_kwargs,
                           orth_init,
                           device)


def build_value_model(env_info: dict,
                      module_config: net.ModuleConfig,
                      dtype: str,
                      orth_init: bool,
                      device: str):
    """
    ������ֵ����
    :param env_info:
    :param module_config:
    :param dtype:
    :param orth_init:
    :param device:
    :return:
    """
    # mlp�ṹ
    value_mlp_dim = module_config.value_mlp_dim if module_config.value_mlp_dim is not None \
        else module_config.policy_mlp_dim
    # cnn�ṹ
    value_cnn_param = module_config.value_cnn_param if module_config.value_cnn_param is not None \
        else module_config.policy_cnn_param

    # ������ȡ������
    extractor_kwargs = {
        "state_dim": env_info["state_dim"],
        "action_dim": env_info["action_dim"],
        "dtype": dtype,
        "mlp_dim": value_mlp_dim,
        "cnn_param": value_cnn_param,
        "device": device
    }

    return net.ValueModel(env_info["state_dim"],
                          net.ValueExtractor,
                          extractor_kwargs,
                          orth_init,
                          device)

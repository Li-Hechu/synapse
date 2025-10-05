# -*- coding:gbk -*-


import torch
from .action_dist import *

__all__ = ["ModuleConfig", "RLNet", "BaseExtractor", "PolicyModel", "ValueModel"]


class ModuleConfig(object):
    """
    神经网络配置
    """
    def __init__(self,
                 net_type: str = "mlp",
                 policy_mlp_dim: tuple = (64, 64),
                 policy_cnn_param: dict = None,
                 value_mlp_dim: tuple = None,
                 value_cnn_param: dict = None):
        """
        构造函数
        :param net_type: 网络类型, mlp 或者 cnn
        :param policy_mlp_dim: 策略网络mlp结构
        :param policy_cnn_param: 策略网络cnn参数
        :param value_mlp_dim: 价值网络mlp结构，若为None，则结构同策略网络
        :param value_cnn_param: 价值网络cnn参数， 若为None, 则结构同策略网络

        ===========================
        cnn_param: 卷积神经网络参数
        如果key为conv，表示卷积层，value为字典，表征Conv2d参数
        如果key为pool，表示最大池化层，value为字典，表征MaxPool2d参数
        如果key为act，表示激活层，忽略value参数
        """
        # 参数检查
        if net_type == "cnn":
            assert policy_cnn_param is not None, "未指定CNN结构"
        self.net_type = net_type
        self.policy_mlp_dim = policy_mlp_dim
        self.policy_cnn_param = policy_cnn_param
        self.value_mlp_dim = value_mlp_dim
        self.value_cnn_param = value_cnn_param


class RLNet(torch.nn.Module):
    """
    强化学习网络预定义
    """

    def __init__(self, device: str):
        """
        构造函数
        :param device: 张量设备
        """
        super(RLNet, self).__init__()

        # 输入维度
        self.input_dim = None
        # 输出维度
        self.output_dim = None
        # 张量设备
        self.device: torch.device = torch.device(device)

    @staticmethod
    def build_mlp_layer(input_dim: int, hidden_dim: tuple = (), output_dim: int = None) -> torch.nn.Sequential:
        """
        构建MLP层
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :return:
        """
        if len(hidden_dim) == 0:
            assert output_dim is not None, f"在没有隐藏层时，输出层参数不能为 None"
            mlp_layers = torch.nn.Linear(input_dim, output_dim)
        else:
            # 输入层
            mlp_layers = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim[0]),
                                             torch.nn.ReLU())
            # 隐藏层
            for i in range(1, len(hidden_dim)):
                mlp_layers.append(torch.nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                mlp_layers.append(torch.nn.ReLU())
            # 输出层
            if output_dim is not None:
                mlp_layers.append(torch.nn.Linear(hidden_dim[-1], output_dim))

        return mlp_layers

    @staticmethod
    def build_cnn_layer(cnn_param: dict) -> torch.nn.Sequential:
        """
        建立CNN层
        :param cnn_param: CNN参数
        :return:
        """
        cnn_layer = torch.nn.Sequential()
        for key, value in cnn_param.items():
            if key.startswith('conv'):
                cnn_layer.append(torch.nn.Conv2d(**value))
            elif key.startswith('pool'):
                cnn_layer.append(torch.nn.MaxPool2d(**value))
            elif key.startswith('act'):
                cnn_layer.append(torch.nn.ReLU())
            else:
                assert f"不支持的参数类型 {key}，键类型需要为 conv 或 pool 或 act"
        return cnn_layer

    @staticmethod
    def get_flatten_dim(model: torch.nn.Module, state_dim: tuple) -> int:
        # 获取展平后的维度
        with torch.no_grad():
            # 构造输入
            state = torch.randn(state_dim)
            # 获取展平维度
            flatten_dim = model(state).view(-1).shape[0]
        return flatten_dim

    @staticmethod
    def orthogonal_init(module: torch.nn.Module, gain: float = 1) -> None:
        """
        正交初始化，用于同策算法
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


class BaseExtractor(RLNet):
    """
    特征提取层基类
    """
    def __init__(self, device: str):
        """
        构造函数
        :param device: 张量设备
        """
        super(BaseExtractor, self).__init__(device)

    def forward(self, *args, **kwargs):
        """
        前向传递
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class PolicyModel(RLNet):
    """
    策略网络
    """
    def __init__(self,
                 state_dim: int | tuple,
                 action_dim: int,
                 bounds: tuple,
                 is_discrete: bool,
                 deterministic: bool,
                 extractor_class: BaseExtractor,
                 extractor_kwargs: dict,
                 orth_init: bool,
                 device: str):
        """
        构造函数
        :param state_dim:
        :param action_dim:
        :param bounds:
        :param is_discrete:
        :param deterministic:
        :param extractor_class:
        :param extractor_kwargs:
        :param orth_init:
        :param device:
        """
        super(PolicyModel, self).__init__(device)

        # 输入输出维度
        self.input_dim = state_dim
        self.output_dim = action_dim
        # 是否为离散动作
        self.is_discrete = is_discrete

        # 特征提取层
        self.extractor = extractor_class(**extractor_kwargs)
        # 动作输出网络
        self.action_net = torch.nn.Linear(self.extractor.output_dim, self.output_dim)
        self.log_std_net = None
        # 动作分布
        self.dist = None

        # 动作分布
        if is_discrete:
            self.dist = CategoricalDistribution(action_dim, device)
        else:
            if not deterministic:
                self.dist = NormalDistribution(action_dim, bounds, device)
                self.log_std_net = torch.nn.Linear(self.extractor.output_dim, self.output_dim)
            else:
                self.dist = DeterministicDistribution(action_dim, bounds, device)

        # 正交初始化
        if orth_init:
            self.orthogonal_init(self.extractor, gain=2 ** 0.5)
            self.orthogonal_init(self.action_net, gain=0.01)

        # 转移到张量设备
        self.to(self.device)

    def forward(self, state: torch.tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        前向传递
        :param state: 当前状态
        :return:
        """
        # 经过特征提取层
        latent_policy = self.extractor(state)

        # 动作输出
        action = self.action_net(latent_policy)

        # 输出分布参数
        if self.log_std_net is not None:
            log_std = self.log_std_net(latent_policy)
            return action, log_std

        # 输出确定性动作
        return action

    def full_pass(self, state: torch.tensor):
        """
        采样动作，获取概率、熵以及最佳动作
        :param state:
        :return:
        """
        # 前向传递
        output = self.forward(state)
        # 更新分布
        if isinstance(self.dist, NormalDistribution):
            self.dist.update_distribution(output[0], output[1])
        else:
            self.dist.update_distribution(output)

        # 采样
        action, log_prob = self.dist.sample()
        # 熵
        entropy = self.dist.entropy()
        # 最佳动作
        best_action = self.dist.mode()

        return action, log_prob, entropy, best_action

    def evaluate_actions(self, state: torch.tensor, action: torch.tensor):
        """
        评估给定动作的对数概率和熵
        :param state:
        :param action:
        :return:
        """
        # 前向传递
        output = self.forward(state)
        # 更新分布
        if isinstance(self.dist, NormalDistribution):
            self.dist.update_distribution(output[0], output[1])
        else:
            self.dist.update_distribution(output)

        # 如果是离散动作则变为整数形式，并且压缩最后一个维度
        if self.is_discrete:
            assert action.shape[-1] == 1, f"要求动作最后一个维度为1，实际为 {action.shape[-1]}"
            action = action.long().squeeze(-1)

        # 计算给定动作的对数概率
        log_prob = self.dist.log_prob(action)
        # 计算熵
        entropy = self.dist.entropy()

        return log_prob, entropy


class ValueModel(RLNet):
    """
    价值网络
    """
    def __init__(self,
                 state_dim,
                 extractor_class: BaseExtractor,
                 extractor_kwargs: dict,
                 orth_init: bool,
                 device):
        """
        构造函数
        :param state_dim:
        :param extractor_class:
        :param extractor_kwargs:
        :param orth_init:
        :param device:
        """
        super(ValueModel, self).__init__(device)

        # 输入输出维度
        self.input_dim = state_dim
        self.output_dim = 1

        # 特征提取层
        self.extractor = extractor_class(**extractor_kwargs)
        # 输出层
        self.value_net = torch.nn.Linear(self.extractor.output_dim, self.output_dim)

        # 正交初始化
        if orth_init:
            self.orthogonal_init(self.extractor, gain=2 ** 0.5)
            self.orthogonal_init(self.value_net, gain=1)

        # 转移到张量设备
        self.to(self.device)

    def forward(self, state: torch.tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        前向传递
        :param state:
        :param action:
        :return:
        """
        # 特征提取
        latent_value = self.extractor(state, action)
        # 输出价值
        value = self.value_net(latent_value)

        return value

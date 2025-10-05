# -*- coding:gbk -*-


import torch
from .action_dist import *

__all__ = ["ModuleConfig", "RLNet", "BaseExtractor", "PolicyModel", "ValueModel"]


class ModuleConfig(object):
    """
    ����������
    """
    def __init__(self,
                 net_type: str = "mlp",
                 policy_mlp_dim: tuple = (64, 64),
                 policy_cnn_param: dict = None,
                 value_mlp_dim: tuple = None,
                 value_cnn_param: dict = None):
        """
        ���캯��
        :param net_type: ��������, mlp ���� cnn
        :param policy_mlp_dim: ��������mlp�ṹ
        :param policy_cnn_param: ��������cnn����
        :param value_mlp_dim: ��ֵ����mlp�ṹ����ΪNone����ṹͬ��������
        :param value_cnn_param: ��ֵ����cnn������ ��ΪNone, ��ṹͬ��������

        ===========================
        cnn_param: ������������
        ���keyΪconv����ʾ����㣬valueΪ�ֵ䣬����Conv2d����
        ���keyΪpool����ʾ���ػ��㣬valueΪ�ֵ䣬����MaxPool2d����
        ���keyΪact����ʾ����㣬����value����
        """
        # �������
        if net_type == "cnn":
            assert policy_cnn_param is not None, "δָ��CNN�ṹ"
        self.net_type = net_type
        self.policy_mlp_dim = policy_mlp_dim
        self.policy_cnn_param = policy_cnn_param
        self.value_mlp_dim = value_mlp_dim
        self.value_cnn_param = value_cnn_param


class RLNet(torch.nn.Module):
    """
    ǿ��ѧϰ����Ԥ����
    """

    def __init__(self, device: str):
        """
        ���캯��
        :param device: �����豸
        """
        super(RLNet, self).__init__()

        # ����ά��
        self.input_dim = None
        # ���ά��
        self.output_dim = None
        # �����豸
        self.device: torch.device = torch.device(device)

    @staticmethod
    def build_mlp_layer(input_dim: int, hidden_dim: tuple = (), output_dim: int = None) -> torch.nn.Sequential:
        """
        ����MLP��
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :return:
        """
        if len(hidden_dim) == 0:
            assert output_dim is not None, f"��û�����ز�ʱ��������������Ϊ None"
            mlp_layers = torch.nn.Linear(input_dim, output_dim)
        else:
            # �����
            mlp_layers = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim[0]),
                                             torch.nn.ReLU())
            # ���ز�
            for i in range(1, len(hidden_dim)):
                mlp_layers.append(torch.nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                mlp_layers.append(torch.nn.ReLU())
            # �����
            if output_dim is not None:
                mlp_layers.append(torch.nn.Linear(hidden_dim[-1], output_dim))

        return mlp_layers

    @staticmethod
    def build_cnn_layer(cnn_param: dict) -> torch.nn.Sequential:
        """
        ����CNN��
        :param cnn_param: CNN����
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
                assert f"��֧�ֵĲ������� {key}����������ҪΪ conv �� pool �� act"
        return cnn_layer

    @staticmethod
    def get_flatten_dim(model: torch.nn.Module, state_dim: tuple) -> int:
        # ��ȡչƽ���ά��
        with torch.no_grad():
            # ��������
            state = torch.randn(state_dim)
            # ��ȡչƽά��
            flatten_dim = model(state).view(-1).shape[0]
        return flatten_dim

    @staticmethod
    def orthogonal_init(module: torch.nn.Module, gain: float = 1) -> None:
        """
        ������ʼ��������ͬ���㷨
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


class BaseExtractor(RLNet):
    """
    ������ȡ�����
    """
    def __init__(self, device: str):
        """
        ���캯��
        :param device: �����豸
        """
        super(BaseExtractor, self).__init__(device)

    def forward(self, *args, **kwargs):
        """
        ǰ�򴫵�
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class PolicyModel(RLNet):
    """
    ��������
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
        ���캯��
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

        # �������ά��
        self.input_dim = state_dim
        self.output_dim = action_dim
        # �Ƿ�Ϊ��ɢ����
        self.is_discrete = is_discrete

        # ������ȡ��
        self.extractor = extractor_class(**extractor_kwargs)
        # �����������
        self.action_net = torch.nn.Linear(self.extractor.output_dim, self.output_dim)
        self.log_std_net = None
        # �����ֲ�
        self.dist = None

        # �����ֲ�
        if is_discrete:
            self.dist = CategoricalDistribution(action_dim, device)
        else:
            if not deterministic:
                self.dist = NormalDistribution(action_dim, bounds, device)
                self.log_std_net = torch.nn.Linear(self.extractor.output_dim, self.output_dim)
            else:
                self.dist = DeterministicDistribution(action_dim, bounds, device)

        # ������ʼ��
        if orth_init:
            self.orthogonal_init(self.extractor, gain=2 ** 0.5)
            self.orthogonal_init(self.action_net, gain=0.01)

        # ת�Ƶ������豸
        self.to(self.device)

    def forward(self, state: torch.tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        ǰ�򴫵�
        :param state: ��ǰ״̬
        :return:
        """
        # ����������ȡ��
        latent_policy = self.extractor(state)

        # �������
        action = self.action_net(latent_policy)

        # ����ֲ�����
        if self.log_std_net is not None:
            log_std = self.log_std_net(latent_policy)
            return action, log_std

        # ���ȷ���Զ���
        return action

    def full_pass(self, state: torch.tensor):
        """
        ������������ȡ���ʡ����Լ���Ѷ���
        :param state:
        :return:
        """
        # ǰ�򴫵�
        output = self.forward(state)
        # ���·ֲ�
        if isinstance(self.dist, NormalDistribution):
            self.dist.update_distribution(output[0], output[1])
        else:
            self.dist.update_distribution(output)

        # ����
        action, log_prob = self.dist.sample()
        # ��
        entropy = self.dist.entropy()
        # ��Ѷ���
        best_action = self.dist.mode()

        return action, log_prob, entropy, best_action

    def evaluate_actions(self, state: torch.tensor, action: torch.tensor):
        """
        �������������Ķ������ʺ���
        :param state:
        :param action:
        :return:
        """
        # ǰ�򴫵�
        output = self.forward(state)
        # ���·ֲ�
        if isinstance(self.dist, NormalDistribution):
            self.dist.update_distribution(output[0], output[1])
        else:
            self.dist.update_distribution(output)

        # �������ɢ�������Ϊ������ʽ������ѹ�����һ��ά��
        if self.is_discrete:
            assert action.shape[-1] == 1, f"Ҫ�������һ��ά��Ϊ1��ʵ��Ϊ {action.shape[-1]}"
            action = action.long().squeeze(-1)

        # ������������Ķ�������
        log_prob = self.dist.log_prob(action)
        # ������
        entropy = self.dist.entropy()

        return log_prob, entropy


class ValueModel(RLNet):
    """
    ��ֵ����
    """
    def __init__(self,
                 state_dim,
                 extractor_class: BaseExtractor,
                 extractor_kwargs: dict,
                 orth_init: bool,
                 device):
        """
        ���캯��
        :param state_dim:
        :param extractor_class:
        :param extractor_kwargs:
        :param orth_init:
        :param device:
        """
        super(ValueModel, self).__init__(device)

        # �������ά��
        self.input_dim = state_dim
        self.output_dim = 1

        # ������ȡ��
        self.extractor = extractor_class(**extractor_kwargs)
        # �����
        self.value_net = torch.nn.Linear(self.extractor.output_dim, self.output_dim)

        # ������ʼ��
        if orth_init:
            self.orthogonal_init(self.extractor, gain=2 ** 0.5)
            self.orthogonal_init(self.value_net, gain=1)

        # ת�Ƶ������豸
        self.to(self.device)

    def forward(self, state: torch.tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        ǰ�򴫵�
        :param state:
        :param action:
        :return:
        """
        # ������ȡ
        latent_value = self.extractor(state, action)
        # �����ֵ
        value = self.value_net(latent_value)

        return value

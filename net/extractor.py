# -*- coding:gbk -*-

import torch
from .rlnet import *

__all__ = ["PolicyExtractor", "ValueExtractor"]


class PolicyExtractor(BaseExtractor):
    """
    策略提取网络
    """
    def __init__(self,
                 state_dim,
                 mlp_dim: tuple,
                 cnn_param: dict,
                 device: str
                 ):
        """
        构造函数
        :param state_dim:
        :param mlp_dim:
        :param cnn_param:
        :param device:
        """
        super(PolicyExtractor, self).__init__(device)

        assert mlp_dim is not None, "至少需要给定mlp层参数才可以创建网络"
        assert len(mlp_dim) > 0, "至少给定一层mlp参数才可以创建网络"
        if cnn_param is not None:
            assert len(cnn_param) > 0, "至少给定一层cnn参数才可以创建网络"

        self.input_dim = state_dim
        self.output_dim = mlp_dim[-1]

        # 网络层
        self.mlp_layer = None
        self.cnn_layer = None
        # 仅为MLP网络
        if cnn_param is None:
            # 构造全连接层
            self.mlp_layer = self.build_mlp_layer(state_dim, mlp_dim)
        # 为CNN网络
        else:
            self.cnn_layer = self.build_cnn_layer(cnn_param)
            self.mlp_layer = self.build_mlp_layer(self.get_flatten_dim(self.cnn_layer, state_dim), mlp_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传递
        :param state:
        :return:
        """
        x = state

        # 卷积层
        if self.cnn_layer is not None:
            x = self.cnn_layer(x)
            # 展平
            if len(state.shape) == 3:
                x = x.view(-1)
            elif len(state.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                assert f"输入的状态维度为应当为3或4，实际为 {len(state.shape)}"

        # 全连接层
        x = self.mlp_layer(x)

        return x


class ValueExtractor(BaseExtractor):
    """
    价值提取网络
    """
    def __init__(self,
                 state_dim: int | tuple,
                 action_dim: int,
                 dtype: str,
                 mlp_dim: tuple,
                 cnn_param: dict,
                 device: str
                 ):
        """
        构造函数
        :param state_dim:
        :param action_dim:
        :param dtype:
        :param mlp_dim:
        :param cnn_param:
        :param device:
        """
        super(ValueExtractor, self).__init__(device)

        assert mlp_dim is not None, "至少需要给定mlp层参数才可以创建网络"
        assert len(mlp_dim) > 0, "至少给定一层mlp参数才可以创建网络"
        if cnn_param is not None:
            assert len(cnn_param) > 0, "至少给定一层cnn参数才可以创建网络"
        assert dtype == 'q' or dtype == 'v', f"价值网络类型仅有 q 或 v，但是给定 f{dtype}"

        # 价值类型
        self.dtype = dtype

        # 输入输出维度
        if cnn_param is None:
            self.input_dim = state_dim + (action_dim if dtype == 'q' else 0)
        else:
            self.input_dim = state_dim
        self.output_dim = mlp_dim[-1]

        # 网络层
        self.mlp_layer = None
        self.cnn_layer = None
        # 为MLP网络
        if cnn_param is None:
            # 构造全连接层
            self.mlp_layer = self.build_mlp_layer(self.input_dim, mlp_dim)
        # 为CNN网络
        else:
            self.cnn_layer = self.build_cnn_layer(cnn_param)
            flatten_dim = self.get_flatten_dim(self.cnn_layer, state_dim) + (action_dim if dtype == 'q' else 0)
            self.mlp_layer = self.build_mlp_layer(flatten_dim, mlp_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        前向传递
        :param state:
        :param action:
        :return:
        """
        x = state

        # 卷积层
        if self.cnn_layer is not None:
            x = self.cnn_layer(x)
            # 展平
            if len(state.shape) == 3:
                x = x.view(-1)
            elif len(state.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                assert f"输入的状态维度为应当为3或4，实际为 {len(state.shape)}"

        # 拼接
        if self.dtype == 'q':
            assert action is not None, f"动作价值网络需要给定action，实际为 None"
            x = torch.concat((x, action), dim=-1)

        # 输出
        x = self.mlp_layer(x)

        return x

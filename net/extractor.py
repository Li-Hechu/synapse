# -*- coding:gbk -*-

import torch
from .rlnet import *

__all__ = ["PolicyExtractor", "ValueExtractor"]


class PolicyExtractor(BaseExtractor):
    """
    ������ȡ����
    """
    def __init__(self,
                 state_dim,
                 mlp_dim: tuple,
                 cnn_param: dict,
                 device: str
                 ):
        """
        ���캯��
        :param state_dim:
        :param mlp_dim:
        :param cnn_param:
        :param device:
        """
        super(PolicyExtractor, self).__init__(device)

        assert mlp_dim is not None, "������Ҫ����mlp������ſ��Դ�������"
        assert len(mlp_dim) > 0, "���ٸ���һ��mlp�����ſ��Դ�������"
        if cnn_param is not None:
            assert len(cnn_param) > 0, "���ٸ���һ��cnn�����ſ��Դ�������"

        self.input_dim = state_dim
        self.output_dim = mlp_dim[-1]

        # �����
        self.mlp_layer = None
        self.cnn_layer = None
        # ��ΪMLP����
        if cnn_param is None:
            # ����ȫ���Ӳ�
            self.mlp_layer = self.build_mlp_layer(state_dim, mlp_dim)
        # ΪCNN����
        else:
            self.cnn_layer = self.build_cnn_layer(cnn_param)
            self.mlp_layer = self.build_mlp_layer(self.get_flatten_dim(self.cnn_layer, state_dim), mlp_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        ǰ�򴫵�
        :param state:
        :return:
        """
        x = state

        # �����
        if self.cnn_layer is not None:
            x = self.cnn_layer(x)
            # չƽ
            if len(state.shape) == 3:
                x = x.view(-1)
            elif len(state.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                assert f"�����״̬ά��ΪӦ��Ϊ3��4��ʵ��Ϊ {len(state.shape)}"

        # ȫ���Ӳ�
        x = self.mlp_layer(x)

        return x


class ValueExtractor(BaseExtractor):
    """
    ��ֵ��ȡ����
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
        ���캯��
        :param state_dim:
        :param action_dim:
        :param dtype:
        :param mlp_dim:
        :param cnn_param:
        :param device:
        """
        super(ValueExtractor, self).__init__(device)

        assert mlp_dim is not None, "������Ҫ����mlp������ſ��Դ�������"
        assert len(mlp_dim) > 0, "���ٸ���һ��mlp�����ſ��Դ�������"
        if cnn_param is not None:
            assert len(cnn_param) > 0, "���ٸ���һ��cnn�����ſ��Դ�������"
        assert dtype == 'q' or dtype == 'v', f"��ֵ�������ͽ��� q �� v�����Ǹ��� f{dtype}"

        # ��ֵ����
        self.dtype = dtype

        # �������ά��
        if cnn_param is None:
            self.input_dim = state_dim + (action_dim if dtype == 'q' else 0)
        else:
            self.input_dim = state_dim
        self.output_dim = mlp_dim[-1]

        # �����
        self.mlp_layer = None
        self.cnn_layer = None
        # ΪMLP����
        if cnn_param is None:
            # ����ȫ���Ӳ�
            self.mlp_layer = self.build_mlp_layer(self.input_dim, mlp_dim)
        # ΪCNN����
        else:
            self.cnn_layer = self.build_cnn_layer(cnn_param)
            flatten_dim = self.get_flatten_dim(self.cnn_layer, state_dim) + (action_dim if dtype == 'q' else 0)
            self.mlp_layer = self.build_mlp_layer(flatten_dim, mlp_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        ǰ�򴫵�
        :param state:
        :param action:
        :return:
        """
        x = state

        # �����
        if self.cnn_layer is not None:
            x = self.cnn_layer(x)
            # չƽ
            if len(state.shape) == 3:
                x = x.view(-1)
            elif len(state.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                assert f"�����״̬ά��ΪӦ��Ϊ3��4��ʵ��Ϊ {len(state.shape)}"

        # ƴ��
        if self.dtype == 'q':
            assert action is not None, f"������ֵ������Ҫ����action��ʵ��Ϊ None"
            x = torch.concat((x, action), dim=-1)

        # ���
        x = self.mlp_layer(x)

        return x

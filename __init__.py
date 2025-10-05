# -*-  coding:gbk -*-

"""
Synapse
===============
��������ǿ��ѧϰѵ�����

��Ҫ����
==============
�������õĵ��������㷨
1. ���ڼ�ֵ�ķ���
    DQN
    Dueling DQN
2. ����Actor-Critic�ķ���
    DDPG
    TD3
    SAC
3. �����ݶȷ���
    REINFORCE
    VPG
    A3C
    A2C
    PPO

����
=============
��մ���2025-09-24

������
============
torch==2.8.0
numpy==2.3.3
gymnasium==1.2.0
matplotlib==3.10.6

ʾ��
===========

Example 1. ʹ��SACѵ��HalfCheetah
-------------------------------

import synapse
import gymnasium

# ��������������ʹ�ð�װ����¼ѵ������
env = gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("HalfCheetah-v5", render_mode=None, ctrl_cost_weight=0.1), buffer_length=1000)
# ����������
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# ����ѵ��ģ��
model = synapse.SAC(env,
                    module_config=module_config,
                    lr=0.0005,
                    buffer_size=100000,
                    batch_size=128)
# ģ��ѧϰ��ѧϰ����Ϊ1000��������������Ϊ50
model.learn(1000, 50)
# ģ�Ͳ���
model.evaluate()
# ͳ��ѵ�����ݣ�����ÿ�ֵ��ܽ����������ܲ����Լ�ÿ�ֽ�����ʱ��
synapse.show_episode_statistics(env, window=10)
# �ر�ģ��
model.shut_down()


Example 2. ʹ��VPGѵ��CartPole
-----------------------------

import synapse
import gymnasium

# ��������������ʹ�ð�װ����¼ѵ������
env = gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("CartPole-v1", render_mode='rgb_array'), buffer_length=500)
# ����������
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# ����ѵ��ģ��
model = synapse.VPG(env,
                    module_config=module_config,
                    lr=0.0005)
# ģ��ѧϰ��ѧϰ����Ϊ500��������������Ϊ50
model.learn(500, 50)
# ģ�Ͳ���
model.evaluate()
# ͳ��ѵ�����ݣ�����ÿ�ֵ��ܽ����������ܲ����Լ�ÿ�ֽ�����ʱ��
synapse.show_episode_statistics(env, window=10)
# �ر�ģ��
model.shut_down()


Example 3. ʹ��A3Cѵ��CartPole
----------------------------

import synapse
import gymnasium

# ������������
def env_build_handle():
    return gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("CartPole-v1", render_mode='rgb_array'), buffer_length=500)
# ����������
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# ����ѵ��ģ��
model = synapse.A3C(env_build_handle,
                    module_config=module_config,
                    lr=0.0005,
                    bootstrap_steps=10,
                    worker_num=6)
# ģ��ѧϰ��ѧϰ����Ϊ500��������������Ϊ50
model.learn(500, 50)
# ģ�Ͳ���
model.evaluate()
# ͳ��ѵ�����ݣ�����ÿ�ֵ��ܽ����������ܲ����Լ�ÿ�ֽ�����ʱ��
synapse.show_episode_statistics(env, window=10)
# �ر�ģ��
model.shut_down()


Example 4. ʹ��PPOѵ��CartPole
----------------------------

import synapse
import gymnasium

# ������������
def env_build_handle():
    return gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("CartPole-v1", render_mode='rgb_array'), buffer_length=500)
# �������л�������������Ϊ6
env = gymnasium.vector.SyncVectorEnv([env_build_handle for _ in range(6)])
# ����������
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# ����ѵ��ģ��
model = synapse.PPO(env,
                    module_config=module_config,
                    lr=0.0005,
                    bootstrap_steps=10,
                    buffer_size=500,
                    batch_size=64)
# ģ��ѧϰ��ѧϰ����Ϊ500��������������Ϊ50
model.learn(500, 50)
# ģ�Ͳ���
model.evaluate()
# ͳ��ѵ�����ݣ�����ÿ�ֵ��ܽ����������ܲ����Լ�ÿ�ֽ�����ʱ��
synapse.show_episode_statistics(env.envs[0], window=10)
# �ر�ģ��
model.shut_down()
"""

__author__ = "LHC"
__email__ = "1277813766@qq.com"
__version__ = "1.0.0"
__date__ = "2025-09-24"

from .framework import *
from .sarl import *
from .utils import *
from .net import *

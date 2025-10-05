# -*-  coding:gbk -*-

"""
Synapse
===============
单智能体强化学习训练框架

主要功能
==============
包含常用的单智能体算法
1. 基于价值的方法
    DQN
    Dueling DQN
2. 基于Actor-Critic的方法
    DDPG
    TD3
    SAC
3. 策略梯度方法
    REINFORCE
    VPG
    A3C
    A2C
    PPO

作者
=============
李赫矗，2025-09-24

依赖项
============
torch==2.8.0
numpy==2.3.3
gymnasium==1.2.0
matplotlib==3.10.6

示例
===========

Example 1. 使用SAC训练HalfCheetah
-------------------------------

import synapse
import gymnasium

# 创建环境，并且使用包装器记录训练数据
env = gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("HalfCheetah-v5", render_mode=None, ctrl_cost_weight=0.1), buffer_length=1000)
# 神经网络配置
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# 创建训练模型
model = synapse.SAC(env,
                    module_config=module_config,
                    lr=0.0005,
                    buffer_size=100000,
                    batch_size=128)
# 模型学习，学习步长为1000，环境重置种子为50
model.learn(1000, 50)
# 模型测试
model.evaluate()
# 统计训练数据，包括每轮的总奖励，交互总步数以及每轮交互的时长
synapse.show_episode_statistics(env, window=10)
# 关闭模型
model.shut_down()


Example 2. 使用VPG训练CartPole
-----------------------------

import synapse
import gymnasium

# 创建环境，并且使用包装器记录训练数据
env = gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("CartPole-v1", render_mode='rgb_array'), buffer_length=500)
# 神经网络配置
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# 创建训练模型
model = synapse.VPG(env,
                    module_config=module_config,
                    lr=0.0005)
# 模型学习，学习步长为500，环境重置种子为50
model.learn(500, 50)
# 模型测试
model.evaluate()
# 统计训练数据，包括每轮的总奖励，交互总步数以及每轮交互的时长
synapse.show_episode_statistics(env, window=10)
# 关闭模型
model.shut_down()


Example 3. 使用A3C训练CartPole
----------------------------

import synapse
import gymnasium

# 创建环境函数
def env_build_handle():
    return gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("CartPole-v1", render_mode='rgb_array'), buffer_length=500)
# 神经网络配置
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# 创建训练模型
model = synapse.A3C(env_build_handle,
                    module_config=module_config,
                    lr=0.0005,
                    bootstrap_steps=10,
                    worker_num=6)
# 模型学习，学习步长为500，环境重置种子为50
model.learn(500, 50)
# 模型测试
model.evaluate()
# 统计训练数据，包括每轮的总奖励，交互总步数以及每轮交互的时长
synapse.show_episode_statistics(env, window=10)
# 关闭模型
model.shut_down()


Example 4. 使用PPO训练CartPole
----------------------------

import synapse
import gymnasium

# 创建环境函数
def env_build_handle():
    return gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make("CartPole-v1", render_mode='rgb_array'), buffer_length=500)
# 创建并行环境，并行数量为6
env = gymnasium.vector.SyncVectorEnv([env_build_handle for _ in range(6)])
# 神经网络配置
module_config = synapse.ModuleConfig(policy_mlp_dim=(128, 128), value_mlp_dim=(128, 128))
# 创建训练模型
model = synapse.PPO(env,
                    module_config=module_config,
                    lr=0.0005,
                    bootstrap_steps=10,
                    buffer_size=500,
                    batch_size=64)
# 模型学习，学习步长为500，环境重置种子为50
model.learn(500, 50)
# 模型测试
model.evaluate()
# 统计训练数据，包括每轮的总奖励，交互总步数以及每轮交互的时长
synapse.show_episode_statistics(env.envs[0], window=10)
# 关闭模型
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

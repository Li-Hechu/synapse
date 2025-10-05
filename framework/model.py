# -*- coding:gbk -*-

from . import build
import net
import sarl
import utils
import gymnasium
import numpy as np
import torch
from typing import Callable
import multiprocessing as mp
from PIL import Image

__all__ = ["DQN",
           "TD3",
           "DDPG",
           "SAC",
           "REINFORCE",
           "VPG",
           "A3C",
           "A2C",
           "PPO",
           "Model"]


class Model(object):
    """
    强化学习模型
    """

    def __init__(self, device: str):
        """
        构造函数
        """
        # 环境
        self.env = None
        self.env: gymnasium.Env | gymnasium.vector.VectorEnv | gymnasium.Wrapper
        # 智能体
        self.agent = None
        self.agent: sarl.Agent
        # 回放缓冲区
        self.buffer = None
        # 张量设备
        self.device = torch.device(device)

    def learn(self,
              episode: int,
              env_seed=None):
        """
        模型学习
        :param episode:
        :param env_seed:
        :return:
        """
        ...

    def predict(self, state: np.ndarray):
        """
        智能体动作预测
        :param state: 当前状态
        :return:
        """
        return self.agent.select_action(state, a_type="eval")

    def evaluate(self,
                 render_filename: str = 'video.gif',
                 seed=None,
                 options=None
                 ):
        """
        模型测试
        :param render_filename: 渲染文件保存地址，只有当渲染模式为 'rgb_array' 时才有用
        :param seed: 环境重置种子
        :param options: 环境重置其余项
        :return:
        """
        # 记录数据
        action_list = []
        reward_list = []
        # 动画帧
        gif = []

        # 环境重置
        state, _ = self.env.reset(seed=seed, options=options)
        # 环境结束
        done = False

        # 交互
        while not np.any(done):
            # 选择动作
            action = self.agent.select_action(state, a_type="eval")
            # 与环境交互
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # 结束标志
            done = np.logical_or(terminated, truncated)
            # 渲染
            if self.env.render_mode is not None:
                frame = self.env.render()
                if self.env.render_mode == 'rgb_array':
                    if isinstance(frame, tuple):
                        frame = frame[0]
                    gif.append(Image.fromarray(frame))

            # 记录数据
            action_list.append(action)
            reward_list.append(reward)

        if self.env.render_mode == 'rgb_array':
            gif[0].save(render_filename, save_all=True, append_images=gif[1:], duration=100, loop=0)

        return np.array(action_list), np.array(reward_list)

    def save(self, path: str):
        """
        保存模型
        :param path: 模型路径
        :return:
        """
        self.agent.save_policy(path)

    def load(self, path: str):
        """
        加载模型
        :param path: 模型路径
        :return:
        """
        self.agent.load_policy(path)

    def shut_down(self):
        """
        关闭模型
        :return:
        """
        self.env.close()

    @staticmethod
    def generate_net(*args, **kwargs):
        """
        创建神经网络模型
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def get_env_param(env: gymnasium.Env | gymnasium.vector.VectorEnv):
        """
        获取环境类型参数
        :param env: 环境
        :return:
        """
        # 环境类型参数
        state_dim = None
        action_dim = None
        bounds = None
        is_vector = False
        num_envs = 1
        is_discrete = True

        # 获取状态空间和动作空间
        if isinstance(env, gymnasium.Env):
            obs_space = env.observation_space
            action_space = env.action_space
        elif isinstance(env, gymnasium.vector.VectorEnv):
            obs_space = env.single_observation_space
            action_space = env.single_action_space
            is_vector = True
            num_envs = env.num_envs
        else:
            assert f"未受支持的环境类型 {type(env)}，需要为 gymnasium.Env 或 gymnasium.vector.VectorEnv"
            raise NotImplementedError

        # 获取状态维度
        if isinstance(obs_space, gymnasium.spaces.Box):
            state_dim = obs_space.shape[-1] if len(obs_space.shape) == 1 else obs_space.shape
        else:
            assert f"未受支持的观测空间类型 {type(obs_space)}， 需要为 gymnasium.spaces.Box"

        # 获取动作维度
        if isinstance(action_space, gymnasium.spaces.Discrete):
            action_dim = action_space.n
            is_discrete = True
        elif isinstance(action_space, gymnasium.spaces.Box):
            action_dim = action_space.shape[-1]
            is_discrete = False
            bounds = (action_space.low, action_space.high)

        return {"state_dim": state_dim,
                "action_dim": action_dim,
                "bounds": bounds,
                "is_vector": is_vector,
                "num_envs": num_envs,
                "is_discrete": is_discrete}


class OffPolicyModel(Model):
    """
    异策模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 device: str):
        super(OffPolicyModel, self).__init__(device)

        # 环境
        self.env = env
        # 环境信息
        self.env_info = self.get_env_param(self.env)
        # 开始学习需要达到的批数
        self.learning_start_batch = None
        # 动作是否被解释为索引
        self.action_as_index = True

        # 回放缓冲区
        self.buffer: utils.buffer.SimpleReplayBuffer

    @staticmethod
    def generate_net(env_info,
                     module_config: net.ModuleConfig,
                     policy_num: int,
                     value_num: int,
                     deterministic: bool,
                     device: str,
                     ignore_value: bool,
                     policy_build_handle: Callable[..., net.PolicyModel] = None,
                     value_build_handle: Callable[..., net.ValueModel] = None):
        """
        创建异策算法神经网络
        :param env_info:
        :param module_config:
        :param policy_num:
        :param value_num:
        :param deterministic:
        :param device:
        :param ignore_value:
        :param policy_build_handle:
        :param value_build_handle:
        :return:
        """
        policy_group = np.array([None] * policy_num, dtype=net.PolicyModel)
        value_group = np.array([None] * value_num, dtype=net.ValueModel)

        if module_config is not None:
            for i in range(policy_num):
                policy_group[i] = build.build_policy_model(env_info, module_config, deterministic, False, device)
            if not ignore_value:
                for i in range(value_num):
                    value_group[i] = build.build_value_model(env_info, module_config, 'q', False, device)
        else:
            assert policy_build_handle is not None, f"策略创建函数为None，无法创建策略网络"
            for i in range(policy_num):
                policy_group[i] = policy_build_handle()
            if not ignore_value:
                assert value_build_handle is not None, f"价值创建函数为None，无法创建价值网络"
                for i in range(value_num):
                    value_group[i] = value_build_handle()

        return policy_group, value_group

    def learn(self, episode: int, env_seed=None):
        """
        模型学习
        :param episode: 训练轮次
        :param env_seed: 环境重置种子
        :return:
        """
        for i in range(episode):
            # 重置环境
            state, _ = self.env.reset(seed=env_seed)
            # 结束
            done = False

            # 与环境交互
            while not done:
                # 智能体选择动作
                action = self.agent.select_action(state, "train")
                # 与环境交互
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # 结束标志
                done = terminated or truncated
                # 存入回放缓冲区
                self.buffer.store(state, action, next_state, reward, done)

                # 训练智能体
                if len(self.buffer) > self.learning_start_batch * self.buffer.batch_size:
                    self.agent.optimize(self.buffer.sample())

                # 更新
                state = next_state.copy()

            print(f"第 {i} 轮训练完成")


class OnPolicyModel(Model):
    """
    同策模型
    """

    def __init__(self,
                 env: gymnasium.Env | gymnasium.vector.VectorEnv,
                 device: str):
        """
        构造哈桑农户
        :param env: 环境
        """
        super(OnPolicyModel, self).__init__(device)

        # 环境
        self.env = env
        # 环境信息
        self.env_info = self.get_env_param(self.env)
        # 回放缓冲区
        self.buffer: utils.buffer.TrajectoryBuffer
        # 自举步长
        self.bootstrap_step = np.inf

    @staticmethod
    def generate_net(env_info,
                     module_config: net.ModuleConfig,
                     device: str,
                     ignore_value: bool,
                     policy_build_handle: Callable[..., net.PolicyModel] = None,
                     value_build_handle: Callable[..., net.ValueModel] = None):
        """
        创建同策算法神经网络神经网络
        :param env_info:
        :param module_config:
        :param device:
        :param ignore_value:
        :param policy_build_handle:
        :param value_build_handle:
        :return:
        """
        value_model = None

        if module_config is not None:
            policy_model = build.build_policy_model(env_info, module_config, False, True, device)
            if not ignore_value:
                value_model = build.build_value_model(env_info, module_config, 'v', True, device)
        else:
            policy_model = policy_build_handle().to(device)
            if not ignore_value:
                value_model = value_build_handle().to(device)
        return policy_model, value_model

    def learn(self, episode: int, env_seed=None):
        """
        模型学习
        :param episode: 训练轮次
        :param env_seed: 环境重置种子
        :return:
        """
        for i in range(episode):
            # 重置环境
            state, _ = self.env.reset(seed=env_seed)
            done = False
            # 当前步数
            step = 0

            while not done:
                # 步数加1
                step += 1
                # 选择动作
                action, log_pa, entropy = self.agent.select_action(state, "train")
                # 交互
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                # 存储
                self.buffer.store(state, action, reward)
                # 更新状态
                state = next_state.copy()

                # 判断是否到达自举步数
                if done or step % self.bootstrap_step == 0:
                    # 下一时刻状态值
                    v_next = torch.tensor([0], dtype=torch.float32, device=self.device) if done \
                        else self.agent.value_model(torch.tensor(state, dtype=torch.float32, device=self.device))
                    # 更新
                    self.agent.optimize(self.buffer.recall(v_next))
                    # 清理缓冲区
                    self.buffer.clear()
            # 显示
            print(f"第 {i} 轮训练结束")


class DQN(OffPolicyModel):
    """
    DQN模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 q_type: str = 'dqn',
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0,
                 buffer_size: int = 5000,
                 batch_size: int = 64,
                 learning_start_batch: int = 5,
                 prioritized: bool = False,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 beta_inc_step: int = 5000,
                 selector_type: str = "linear",
                 selector_max_val: float = 0.8,
                 selector_min_val: float = 0.1,
                 selector_decay_step: int = 5000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        """
        构造函数
        :param env: 环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param q_type: 算法类型， dqn ddqn
        :param gamma: 折扣系数
        :param lr: 学习率
        :param tau: 目标网络混合参数
        :param update_interval: 网络更新间隔步长
        :param grad_norm_clipping: 梯度裁剪范围
        :param buffer_size: 缓冲区大小
        :param batch_size: 批大小
        :param learning_start_batch: 网络开始学习的批大小
        :param prioritized: 是否采用优先经验回放
        :param alpha: 优先经验回放的TD误差指数
        :param beta: 优先经验回放的权重系数
        :param selector_type: 动作选择器类型 linear exp ucb softmax
        :param selector_max_val: 参数最大值
        :param selector_min_val: 参数最小值
        :param selector_decay_step: 参数衰减步长
        :param device: 张量设备
        """
        super(DQN, self).__init__(env, device)

        # 开始学习需要达到的批数
        self.learning_start_batch = learning_start_batch
        # 动作是否被解释为索引
        self.action_as_index = True
        # 是否为优先回放缓存
        self.prioritized = prioritized

        # Q学习类型
        assert q_type == 'dqn' or q_type == 'ddqn', f"Q学习类型q_type仅支持 dqn 或者 ddqn， 但给定类型 {q_type}"

        # 创建网络
        policy_group, _ = self.generate_net(self.env_info,
                                            module_config,
                                            policy_num=2,
                                            value_num=0,
                                            deterministic=True,
                                            device=device,
                                            ignore_value=True,
                                            policy_build_handle=policy_build_handle,
                                            value_build_handle=None)
        policy_model, target_model = policy_group

        # 创建动作选择器
        selector = build.build_selector(selector_type, policy_model,
                                        max_val=selector_max_val,
                                        min_val=selector_min_val,
                                        decay_step=selector_decay_step)
        # 创建回放缓冲区
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], 1, buffer_size, batch_size,
                                                prioritized=prioritized, alpha=alpha,
                                                beta=beta, beta_inc_step=beta_inc_step)
        # 创建智能体
        self.agent = sarl.QLearningAgent(policy_model,
                                         target_model,
                                         selector=selector,
                                         q_type=q_type,
                                         gamma=gamma,
                                         lr=lr,
                                         tau=tau,
                                         update_interval=update_interval,
                                         grad_norm_clipping=grad_norm_clipping)

    def learn(self, episode: int, env_seed=None):
        """
        模型学习
        :param episode: 训练轮次
        :param env_seed: 环境重置种子
        :return:
        """
        for i in range(episode):
            # 重置环境
            state, _ = self.env.reset(seed=env_seed)
            # 结束
            done = True

            # 与环境交互
            while not done:
                # 智能体选择动作
                action = self.agent.select_action(state, "train")
                # 与环境交互
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # 结束标志
                done = terminated or truncated
                # 存入回放缓冲区
                if not self.prioritized:
                    self.buffer.store(state, action, next_state, reward, done)
                else:
                    if len(self.buffer) > self.learning_start_batch:
                        next_state = torch.tensor(next_state, dtype=torch.float, device=self.agent.device)
                        state = torch.tensor(state, dtype=torch.float, device=self.agent.device)
                        with torch.no_grad():
                            # 计算TD目标
                            td_target = self.agent.calculate_td_target(next_state, reward, float(done))
                            # 计算td误差
                            td_error = td_target - self.agent.policy_model(state)[action.item()]
                    else:
                        # 网络还没有开始训练的时候，给定相同的td误差，使得均匀抽样
                        td_error = 1.0
                    self.buffer.store(td_error.item(), state, action, next_state, reward, done)

                # 训练智能体
                if len(self.buffer) > self.learning_start_batch * self.buffer.batch_size:
                    per_data = self.agent.optimize(self.buffer.sample())
                    if self.prioritized:
                        index, td_error = per_data
                        self.buffer.update(index, td_error)

                # 更新
                state = next_state.copy()

            print(f"第 {i} 轮训练完成")


class DDPG(OffPolicyModel):
    """
    DDPG模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 module_config: net.ModuleConfig,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0,
                 buffer_size: int = 5000,
                 batch_size: int = 64,
                 learning_start_batch: int = 5,
                 selector_sigma_max_val: float = 1.0,
                 selector_sigma_min_val: float = 0.1,
                 selector_sigma_decay_step: int = 5000,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 ):
        """
        DDPG模型
        :param env: 环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param tau: 目标网络混合比例
        :param update_interval: 目标网络更新间隔步长
        :param grad_norm_clipping: 梯度裁剪范围
        :param buffer_size: 缓冲区大小
        :param batch_size: 批大小
        :param learning_start_batch:
        :param selector_sigma_max_val:
        :param selector_sigma_min_val:
        :param selector_sigma_decay_step
        :param device: 张量设备
        """
        super(DDPG, self).__init__(env, device)

        # 开始学习所需要的批数
        self.learning_start_batch = learning_start_batch
        # 动作是否被解释为索引
        self.action_as_index = False

        # 创建网络
        policy_group, value_group = self.generate_net(self.env_info,
                                                      module_config,
                                                      policy_num=2,
                                                      value_num=2,
                                                      deterministic=True,
                                                      device=device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)
        online_policy_model, target_policy_model = policy_group
        online_value_model, target_value_model = value_group

        # 创建动作选择器
        selector = utils.selector.GaussianSelector(selector_sigma_max_val,
                                                   selector_sigma_min_val,
                                                   selector_sigma_decay_step,
                                                   model=online_policy_model,
                                                   action_bounds=self.env_info["bounds"])
        # 创建回放缓冲区
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], self.env_info["action_dim"],
                                                buffer_size, batch_size, prioritized=False,
                                                alpha=None, beta=None)
        # 创建智能体
        self.agent = sarl.DDPGAgent(online_policy_model,
                                    target_policy_model,
                                    online_value_model,
                                    target_value_model,
                                    selector=selector,
                                    gamma=gamma,
                                    lr=lr,
                                    tau=tau,
                                    update_interval=update_interval,
                                    grad_norm_clipping=grad_norm_clipping)


class TD3(OffPolicyModel):
    """
    TD3模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 module_config: net.ModuleConfig,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0,
                 buffer_size: int = 5000,
                 batch_size: int = 64,
                 learning_start_batch: int = 5,
                 target_noise_clipping: float = 1.0,
                 noise_sigma: float = 1.0,
                 selector_sigma_max_val: float = 1.0,
                 selector_sigma_min_val: float = 0.1,
                 selector_sigma_decay_step: int = 5000,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 ):
        """
        构造函数
        :param env: 智能体环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param tau: 目标网络混合系数
        :param update_interval: 更新间隔步长
        :param grad_norm_clipping: 梯度裁剪阈值
        :param buffer_size: 缓冲区大小
        :param batch_size: 批大小
        :param learning_start_batch: 网络开始学习的批大小
        :param target_noise_clipping: 动作噪声裁剪范围
        :param noise_sigma: 动作噪声标准差
        :param selector_sigma_max_val: 动作选择器标准差最大值
        :param selector_sigma_min_val: 动作选择器标准差最大值
        :param selector_sigma_decay_step: 动作选择器标准差衰减步数
        :param device: 张量设备
        """
        super(TD3, self).__init__(env, device)

        # 开始学习所需要的批数
        self.learning_start_batch = learning_start_batch
        # 动作是否被解释为索引
        self.action_as_index = False

        # 创建网络
        policy_group, value_group = self.generate_net(self.env_info,
                                                      module_config,
                                                      policy_num=2,
                                                      value_num=4,
                                                      deterministic=True,
                                                      device=device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)
        online_policy_model, target_policy_model = policy_group
        online_value_model, target_value_model, online_value_model_1, target_value_model_1 = value_group

        # 创建动作选择器
        selector = utils.selector.GaussianSelector(selector_sigma_max_val,
                                                   selector_sigma_min_val,
                                                   selector_sigma_decay_step,
                                                   model=online_policy_model,
                                                   action_bounds=self.env_info["bounds"])
        # 创建回放缓冲区
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], self.env_info["action_dim"],
                                                buffer_size, batch_size, prioritized=False,
                                                alpha=None, beta=None)
        # 创建智能体
        self.agent = sarl.TD3Agent(online_policy_model,
                                   target_policy_model,
                                   online_value_model,
                                   target_value_model,
                                   online_value_model_1,
                                   target_value_model_1,
                                   selector=selector,
                                   gamma=gamma,
                                   lr=lr,
                                   tau=tau,
                                   update_interval=update_interval,
                                   grad_norm_clipping=grad_norm_clipping,
                                   target_noise_clipping=target_noise_clipping,
                                   noise_sigma=noise_sigma)


class SAC(OffPolicyModel):
    """
    SAC模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 target_entropy: float = None,
                 entropy_lr: float = 0.001,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 tau: float = 0.1,
                 update_interval: int = 5,
                 grad_norm_clipping: float = 1.0,
                 buffer_size: int = 5000,
                 batch_size: int = 64,
                 learning_start_batch: int = 5,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 ):
        """
        构造函数
        :param env: 环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param target_entropy: 目标熵
        :param entropy_lr: 熵学习率
        :param gamma: 折扣系数
        :param lr: 学习率
        :param tau: 目标网络更新比率
        :param update_interval: 目标网络更新步长
        :param grad_norm_clipping: 梯度裁剪范围
        :param buffer_size: 缓冲区大小
        :param batch_size: 批大小
        :param learning_start_batch: 开始学习时搜集到的批次
        :param device: 张量设备
        """
        super(SAC, self).__init__(env, device)

        # 开始学习所需要的批数
        self.learning_start_batch = learning_start_batch
        # 动作是否被解释为索引
        self.action_as_index = False
        # 目标熵
        if target_entropy is None:
            target_entropy = -np.prod(self.env.action_space.shape).item()

        # 创建网络
        policy_group, value_group = self.generate_net(self.env_info,
                                                      module_config,
                                                      policy_num=1,
                                                      value_num=4,
                                                      deterministic=False,
                                                      device=device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)
        policy_model, _ = policy_group
        online_value_model_a, target_value_model_a, online_value_model_b, target_value_model_b = value_group

        # 创建回放缓冲区
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], self.env_info["action_dim"],
                                                buffer_size, batch_size, prioritized=False,
                                                alpha=None, beta=None)
        # 创建智能体
        self.agent = sarl.SACAgent(policy_model,
                                   online_value_model_a,
                                   target_value_model_a,
                                   online_value_model_b,
                                   target_value_model_b,
                                   target_entropy=target_entropy,
                                   entropy_lr=entropy_lr,
                                   gamma=gamma,
                                   lr=lr,
                                   tau=tau,
                                   update_interval=update_interval,
                                   grad_norm_clipping=grad_norm_clipping)


class REINFORCE(OnPolicyModel):
    """
    REINFORCE模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 grad_norm_clipping: float = 1.0,
                 trajectory_max_length: int = 2000,
                 device="cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """
        构造函数
        :param env: 环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param trajectory_max_length: 单条轨迹最大长度
        :param device: 张量设备
        """
        super(REINFORCE, self).__init__(env, device)

        # 创建网络
        policy_model, _ = self.generate_net(self.env_info,
                                            module_config,
                                            device,
                                            ignore_value=True,
                                            policy_build_handle=policy_build_handle,
                                            value_build_handle=None)
        # 轨迹缓冲区
        self.buffer = utils.buffer.TrajectoryBuffer(self.env_info["state_dim"], 1,
                                                    gamma=gamma, capacity=trajectory_max_length, device=device)
        # 智能体
        self.agent = sarl.REINFORCEAgent(policy_model,
                                         gamma=gamma,
                                         lr=lr,
                                         grad_norm_clipping=grad_norm_clipping)


class VPG(OnPolicyModel):
    """
    VPG模型
    """

    def __init__(self,
                 env: gymnasium.Env,
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 grad_norm_clipping: float = 1.0,
                 entropy_loss_weight: float = 0.01,
                 trajectory_max_length: int = 2000,
                 device="cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """
        VPG模型
        :param env: 环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param entropy_loss_weight: 熵损失权重
        :param trajectory_max_length: 单条轨迹最大长度
        :param device: 张量设备
        """
        super(VPG, self).__init__(env, device)

        # 创建网络
        policy_model, value_model = self.generate_net(self.env_info,
                                                      module_config,
                                                      device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)

        # 轨迹缓冲区
        self.buffer = utils.buffer.TrajectoryBuffer(self.env_info["state_dim"], 1,
                                                    gamma=gamma, capacity=trajectory_max_length, device=device)
        # 智能体
        self.agent = sarl.VPGAgent(policy_model,
                                   value_model,
                                   gamma=gamma,
                                   lr=lr,
                                   grad_norm_clipping=grad_norm_clipping,
                                   entropy_loss_weight=entropy_loss_weight)


class A3C(OnPolicyModel):
    """
    A3C模型
    """

    def __init__(self,
                 env_build_handle: Callable[..., gymnasium.Env],
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 bootstrap_steps: int = 20,
                 worker_num: int = 5,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 grad_norm_clipping: float = 1.0,
                 entropy_loss_weight: float = 0.01,
                 trajectory_max_length: int = 2000,
                 ):
        """
        构造函数
        :param env_build_handle: 环境创建函数
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param bootstrap_steps: 自举步长
        :param worker_num: 工作器个数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param entropy_loss_weight: 熵损失权重
        :param trajectory_max_length: 轨迹最大长度
        """
        super(A3C, self).__init__(env_build_handle(), "cpu")

        # 创建环境函数
        self.env_build_handle = env_build_handle
        # 自举步长
        self.bootstrap_steps = bootstrap_steps
        # 工作器个数
        self.worker_num = worker_num
        # 工作器列表
        self.worker: list[mp.Process] = []

        # 创建共享网络
        self.shared_policy_model, self.shared_value_model = self.generate_net(self.env_info,
                                                                              module_config,
                                                                              device='cpu',
                                                                              ignore_value=False,
                                                                              policy_build_handle=policy_build_handle,
                                                                              value_build_handle=value_build_handle)
        # 张量设备，强制处于CPU
        self.shared_policy_model.to(self.device)
        self.shared_value_model.to(self.device)
        # 共享内存
        self.shared_policy_model = self.shared_policy_model.share_memory()
        self.shared_value_model = self.shared_value_model.share_memory()
        # 创建共享优化器
        self.shared_policy_optimizer = utils.share_optim.SharedAdam(self.shared_policy_model.parameters(), lr=lr)
        self.shared_value_optimizer = utils.share_optim.SharedAdam(self.shared_value_model.parameters(), lr=lr)

        # 存储参数
        self.worker_args = (self.env_build_handle,
                            self.shared_policy_model,
                            self.shared_value_model,
                            self.shared_policy_optimizer,
                            self.shared_value_optimizer,
                            module_config,
                            policy_build_handle,
                            value_build_handle,
                            bootstrap_steps,
                            gamma,
                            grad_norm_clipping,
                            entropy_loss_weight,
                            trajectory_max_length,
                            )

    @staticmethod
    def a3c_worker(env_build_handle: Callable[..., gymnasium.Env],
                   shared_policy_model,
                   shared_value_model,
                   shared_policy_optimizer,
                   shared_value_optimizer,
                   module_config: net.ModuleConfig = None,
                   policy_build_handle: Callable[..., net.PolicyModel] = None,
                   value_build_handle: Callable[..., net.ValueModel] = None,
                   bootstrap_steps: int = 5,
                   gamma: float = 0.99,
                   grad_norm_clipping: float = 1.0,
                   entropy_loss_weight: float = 0.01,
                   trajectory_max_length: int = 2000,
                   state_dim: tuple | int = None,
                   episode: int = 1000,
                   env_seed=None
                   ):
        """
        A3C工作器
        :param env_build_handle:
        :param shared_policy_model:
        :param shared_value_model:
        :param shared_policy_optimizer:
        :param shared_value_optimizer:
        :param module_config:
        :param policy_build_handle:
        :param value_build_handle:
        :param bootstrap_steps:
        :param gamma:
        :param grad_norm_clipping:
        :param entropy_loss_weight:
        :param trajectory_max_length:
        :param state_dim:
        :param episode:
        :param env_seed:
        :return:
        """
        # 获取当前进程
        cur_process = mp.current_process()
        # 创建环境
        env = env_build_handle()
        # 本地策略网络和价值网络
        policy_model, value_model = OnPolicyModel.generate_net(OnPolicyModel.get_env_param(env),
                                                               module_config,
                                                               device='cpu',
                                                               ignore_value=False,
                                                               policy_build_handle=policy_build_handle,
                                                               value_build_handle=value_build_handle)
        # 轨迹缓冲区
        buffer = utils.buffer.TrajectoryBuffer(state_dim, 1,
                                               gamma=gamma, capacity=trajectory_max_length, device='cpu')
        # 智能体
        agent = sarl.A3CAgent(shared_policy_model,
                              shared_value_model,
                              shared_policy_optimizer,
                              shared_value_optimizer,
                              policy_model,
                              value_model,
                              gamma=gamma,
                              grad_norm_clipping=grad_norm_clipping,
                              entropy_loss_weight=entropy_loss_weight)

        for i in range(episode):
            # 重置环境
            state, _ = env.reset(seed=env_seed)
            done = False
            # 当前步数
            step = 0

            while not done:
                # 选择动作
                action, _, _ = agent.select_action(state, a_type="train")
                # 交互
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # 存储
                buffer.store(state, action, float(reward))
                # 更新状态
                state = next_state.copy()

                # 判断是否到达自举步数
                if done or step % bootstrap_steps == 0:
                    # 下一时刻状态值
                    v_next = torch.zeros([0], dtype=torch.float32, device='cpu') if done \
                        else agent.value_model(torch.tensor(state, dtype=torch.float32, device='cpu'))
                    # 更新
                    agent.optimize(buffer.recall(v_next))
                    # 清理缓冲区
                    buffer.clear()
            # 显示
            print(f"进程 {cur_process.name}, 第 {i} 轮训练结束")

    def learn(self, episode: int, env_seed=None):
        # 各个环境的初始化种子
        if isinstance(env_seed, int) or env_seed is None:
            env_seed = [env_seed for _ in range(self.worker_num)]
        # 获取状态维度

        # 多进程工作器
        self.worker = [mp.Process(target=A3C.a3c_worker, args=(*self.worker_args,
                                                               self.env_info["state_dim"],
                                                               episode,
                                                               env_seed[i]))
                       for i in range(self.worker_num)]
        # 进程开始
        for worker in self.worker:
            worker.start()
        # 等待进程结束
        for worker in self.worker:
            worker.join()


class A2C(OnPolicyModel):
    """
    A2C模型
    """

    def __init__(self,
                 env: gymnasium.vector.VectorEnv,
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 gamma: float = 0.99,
                 lr: float = 0.0001,
                 grad_norm_clipping: float = 1.0,
                 gae_norm: bool = False,
                 gae_lambda: float = 0.95,
                 entropy_loss_weight: float = 0.1,
                 buffer_size: int = 200,
                 device="cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """
        构造函数
        :param env: 并行环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param gae_lambda: GAE的奖励折扣系数
        :param entropy_loss_weight: 熵损失权重
        :param device: 张量设备
        """
        assert isinstance(env, gymnasium.vector.VectorEnv), f"需要传入的是并行环境VecEnv，实际为 {type(env)}"

        super(A2C, self).__init__(env, device)

        # 创建网络
        policy_model, value_model = self.generate_net(self.env_info,
                                                      module_config,
                                                      device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)

        # 智能体
        self.agent = sarl.A2CAgent(policy_model,
                                   value_model,
                                   gamma=gamma,
                                   lr=lr,
                                   grad_norm_clipping=grad_norm_clipping,
                                   gae_lambda=gae_lambda,
                                   entropy_loss_weight=entropy_loss_weight,
                                   gae_norm=gae_norm)

        # 轨迹缓冲区
        self.buffer = utils.buffer.RolloutBuffer(self.env_info["state_dim"], 1,
                                                 self.env,
                                                 buffer_size, None, gamma, gae_lambda, device)

    def learn(self, episode: int, env_seed=None):
        """
        模型学习
        :param episode: 训练轮次
        :param env_seed: 环境种子
        :return:
        """
        for i in range(episode):
            # 训练
            self.agent.optimize(self.buffer, seed=env_seed)
            # 打印训练完成
            print(f"第 {i} 轮训练结束")


class PPO(OnPolicyModel):
    """
    A2C模型
    """

    def __init__(self,
                 env: gymnasium.vector.VectorEnv,
                 module_config: net.ModuleConfig = None,
                 policy_build_handle: Callable[..., net.PolicyModel] = None,
                 value_build_handle: Callable[..., net.ValueModel] = None,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 grad_norm_clipping: float = 1.0,
                 gae_lambda: float = 0.95,
                 gae_norm: bool = False,
                 entropy_loss_weight: float = 0.05,
                 buffer_size: int = 512,
                 batch_size: int = 64,
                 update_epoch: int = 5,
                 policy_clipping: float = 0.2,
                 value_clipping: float = 0.2,
                 device="cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """
        构造函数
        :param env: 并行环境
        :param module_config: 神经网络配置
        :param policy_build_handle: 策略网络创建函数
        :param value_build_handle: 价值网络创建函数
        :param gamma: 折扣系数
        :param lr: 学习率
        :param grad_norm_clipping: 梯度裁剪范围
        :param gae_lambda: GAE的奖励折扣系数
        :param gae_norm: 是否对GAE进行标准化
        :param entropy_loss_weight: 熵损失权重
        :param buffer_size: 轨迹缓冲区大小
        :param batch_size: 采样批大小
        :param update_epoch: 网络重复更新次数
        :param policy_clipping: 策略新旧概率比裁剪范围
        :param value_clipping: 新旧价值之差裁剪范围
        :param device: 张量设备
        """
        assert isinstance(env, gymnasium.vector.VectorEnv), f"需要传入的是并行环境VecEnv，实际为 {type(env)}"

        super(PPO, self).__init__(env, device)

        # 获取状态动作维度


        # 创建网络
        policy_model, value_model = self.generate_net(self.env_info,
                                                      module_config,
                                                      device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)

        # 创建智能体
        self.agent = sarl.PPOAgent(policy_model,
                                   value_model,
                                   gamma=gamma,
                                   lr=lr,
                                   grad_norm_clipping=grad_norm_clipping,
                                   gae_lambda=gae_lambda,
                                   entropy_loss_weight=entropy_loss_weight,
                                   gae_norm=gae_norm,
                                   update_epoch=update_epoch,
                                   policy_clipping=policy_clipping,
                                   value_clipping=value_clipping,
                                   )

        # 轨迹缓冲区
        self.buffer = utils.buffer.RolloutBuffer(self.env_info["state_dim"], 1,
                                                 self.env,
                                                 buffer_size, batch_size, gamma, gae_lambda, device)

    def learn(self, episode: int, env_seed=None):
        """
        模型学习
        :param episode: 训练轮次
        :param env_seed: 环境种子
        :return:
        """
        for i in range(episode):
            # 训练
            self.agent.optimize(self.buffer, seed=env_seed)
            # 打印训练完成
            print(f"第 {i} 轮训练结束")

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
    ǿ��ѧϰģ��
    """

    def __init__(self, device: str):
        """
        ���캯��
        """
        # ����
        self.env = None
        self.env: gymnasium.Env | gymnasium.vector.VectorEnv | gymnasium.Wrapper
        # ������
        self.agent = None
        self.agent: sarl.Agent
        # �طŻ�����
        self.buffer = None
        # �����豸
        self.device = torch.device(device)

    def learn(self,
              episode: int,
              env_seed=None):
        """
        ģ��ѧϰ
        :param episode:
        :param env_seed:
        :return:
        """
        ...

    def predict(self, state: np.ndarray):
        """
        �����嶯��Ԥ��
        :param state: ��ǰ״̬
        :return:
        """
        return self.agent.select_action(state, a_type="eval")

    def evaluate(self,
                 render_filename: str = 'video.gif',
                 seed=None,
                 options=None
                 ):
        """
        ģ�Ͳ���
        :param render_filename: ��Ⱦ�ļ������ַ��ֻ�е���ȾģʽΪ 'rgb_array' ʱ������
        :param seed: ������������
        :param options: ��������������
        :return:
        """
        # ��¼����
        action_list = []
        reward_list = []
        # ����֡
        gif = []

        # ��������
        state, _ = self.env.reset(seed=seed, options=options)
        # ��������
        done = False

        # ����
        while not np.any(done):
            # ѡ����
            action = self.agent.select_action(state, a_type="eval")
            # �뻷������
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # ������־
            done = np.logical_or(terminated, truncated)
            # ��Ⱦ
            if self.env.render_mode is not None:
                frame = self.env.render()
                if self.env.render_mode == 'rgb_array':
                    if isinstance(frame, tuple):
                        frame = frame[0]
                    gif.append(Image.fromarray(frame))

            # ��¼����
            action_list.append(action)
            reward_list.append(reward)

        if self.env.render_mode == 'rgb_array':
            gif[0].save(render_filename, save_all=True, append_images=gif[1:], duration=100, loop=0)

        return np.array(action_list), np.array(reward_list)

    def save(self, path: str):
        """
        ����ģ��
        :param path: ģ��·��
        :return:
        """
        self.agent.save_policy(path)

    def load(self, path: str):
        """
        ����ģ��
        :param path: ģ��·��
        :return:
        """
        self.agent.load_policy(path)

    def shut_down(self):
        """
        �ر�ģ��
        :return:
        """
        self.env.close()

    @staticmethod
    def generate_net(*args, **kwargs):
        """
        ����������ģ��
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def get_env_param(env: gymnasium.Env | gymnasium.vector.VectorEnv):
        """
        ��ȡ�������Ͳ���
        :param env: ����
        :return:
        """
        # �������Ͳ���
        state_dim = None
        action_dim = None
        bounds = None
        is_vector = False
        num_envs = 1
        is_discrete = True

        # ��ȡ״̬�ռ�Ͷ����ռ�
        if isinstance(env, gymnasium.Env):
            obs_space = env.observation_space
            action_space = env.action_space
        elif isinstance(env, gymnasium.vector.VectorEnv):
            obs_space = env.single_observation_space
            action_space = env.single_action_space
            is_vector = True
            num_envs = env.num_envs
        else:
            assert f"δ��֧�ֵĻ������� {type(env)}����ҪΪ gymnasium.Env �� gymnasium.vector.VectorEnv"
            raise NotImplementedError

        # ��ȡ״̬ά��
        if isinstance(obs_space, gymnasium.spaces.Box):
            state_dim = obs_space.shape[-1] if len(obs_space.shape) == 1 else obs_space.shape
        else:
            assert f"δ��֧�ֵĹ۲�ռ����� {type(obs_space)}�� ��ҪΪ gymnasium.spaces.Box"

        # ��ȡ����ά��
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
    ���ģ��
    """

    def __init__(self,
                 env: gymnasium.Env,
                 device: str):
        super(OffPolicyModel, self).__init__(device)

        # ����
        self.env = env
        # ������Ϣ
        self.env_info = self.get_env_param(self.env)
        # ��ʼѧϰ��Ҫ�ﵽ������
        self.learning_start_batch = None
        # �����Ƿ񱻽���Ϊ����
        self.action_as_index = True

        # �طŻ�����
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
        ��������㷨������
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
            assert policy_build_handle is not None, f"���Դ�������ΪNone���޷�������������"
            for i in range(policy_num):
                policy_group[i] = policy_build_handle()
            if not ignore_value:
                assert value_build_handle is not None, f"��ֵ��������ΪNone���޷�������ֵ����"
                for i in range(value_num):
                    value_group[i] = value_build_handle()

        return policy_group, value_group

    def learn(self, episode: int, env_seed=None):
        """
        ģ��ѧϰ
        :param episode: ѵ���ִ�
        :param env_seed: ������������
        :return:
        """
        for i in range(episode):
            # ���û���
            state, _ = self.env.reset(seed=env_seed)
            # ����
            done = False

            # �뻷������
            while not done:
                # ������ѡ����
                action = self.agent.select_action(state, "train")
                # �뻷������
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # ������־
                done = terminated or truncated
                # ����طŻ�����
                self.buffer.store(state, action, next_state, reward, done)

                # ѵ��������
                if len(self.buffer) > self.learning_start_batch * self.buffer.batch_size:
                    self.agent.optimize(self.buffer.sample())

                # ����
                state = next_state.copy()

            print(f"�� {i} ��ѵ�����")


class OnPolicyModel(Model):
    """
    ͬ��ģ��
    """

    def __init__(self,
                 env: gymnasium.Env | gymnasium.vector.VectorEnv,
                 device: str):
        """
        �����ɣũ��
        :param env: ����
        """
        super(OnPolicyModel, self).__init__(device)

        # ����
        self.env = env
        # ������Ϣ
        self.env_info = self.get_env_param(self.env)
        # �طŻ�����
        self.buffer: utils.buffer.TrajectoryBuffer
        # �Ծٲ���
        self.bootstrap_step = np.inf

    @staticmethod
    def generate_net(env_info,
                     module_config: net.ModuleConfig,
                     device: str,
                     ignore_value: bool,
                     policy_build_handle: Callable[..., net.PolicyModel] = None,
                     value_build_handle: Callable[..., net.ValueModel] = None):
        """
        ����ͬ���㷨������������
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
        ģ��ѧϰ
        :param episode: ѵ���ִ�
        :param env_seed: ������������
        :return:
        """
        for i in range(episode):
            # ���û���
            state, _ = self.env.reset(seed=env_seed)
            done = False
            # ��ǰ����
            step = 0

            while not done:
                # ������1
                step += 1
                # ѡ����
                action, log_pa, entropy = self.agent.select_action(state, "train")
                # ����
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                # �洢
                self.buffer.store(state, action, reward)
                # ����״̬
                state = next_state.copy()

                # �ж��Ƿ񵽴��Ծٲ���
                if done or step % self.bootstrap_step == 0:
                    # ��һʱ��״ֵ̬
                    v_next = torch.tensor([0], dtype=torch.float32, device=self.device) if done \
                        else self.agent.value_model(torch.tensor(state, dtype=torch.float32, device=self.device))
                    # ����
                    self.agent.optimize(self.buffer.recall(v_next))
                    # ��������
                    self.buffer.clear()
            # ��ʾ
            print(f"�� {i} ��ѵ������")


class DQN(OffPolicyModel):
    """
    DQNģ��
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
        ���캯��
        :param env: ����
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param q_type: �㷨���ͣ� dqn ddqn
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: Ŀ�������ϲ���
        :param update_interval: ������¼������
        :param grad_norm_clipping: �ݶȲü���Χ
        :param buffer_size: ��������С
        :param batch_size: ����С
        :param learning_start_batch: ���翪ʼѧϰ������С
        :param prioritized: �Ƿ�������Ⱦ���ط�
        :param alpha: ���Ⱦ���طŵ�TD���ָ��
        :param beta: ���Ⱦ���طŵ�Ȩ��ϵ��
        :param selector_type: ����ѡ�������� linear exp ucb softmax
        :param selector_max_val: �������ֵ
        :param selector_min_val: ������Сֵ
        :param selector_decay_step: ����˥������
        :param device: �����豸
        """
        super(DQN, self).__init__(env, device)

        # ��ʼѧϰ��Ҫ�ﵽ������
        self.learning_start_batch = learning_start_batch
        # �����Ƿ񱻽���Ϊ����
        self.action_as_index = True
        # �Ƿ�Ϊ���ȻطŻ���
        self.prioritized = prioritized

        # Qѧϰ����
        assert q_type == 'dqn' or q_type == 'ddqn', f"Qѧϰ����q_type��֧�� dqn ���� ddqn�� ���������� {q_type}"

        # ��������
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

        # ��������ѡ����
        selector = build.build_selector(selector_type, policy_model,
                                        max_val=selector_max_val,
                                        min_val=selector_min_val,
                                        decay_step=selector_decay_step)
        # �����طŻ�����
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], 1, buffer_size, batch_size,
                                                prioritized=prioritized, alpha=alpha,
                                                beta=beta, beta_inc_step=beta_inc_step)
        # ����������
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
        ģ��ѧϰ
        :param episode: ѵ���ִ�
        :param env_seed: ������������
        :return:
        """
        for i in range(episode):
            # ���û���
            state, _ = self.env.reset(seed=env_seed)
            # ����
            done = True

            # �뻷������
            while not done:
                # ������ѡ����
                action = self.agent.select_action(state, "train")
                # �뻷������
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # ������־
                done = terminated or truncated
                # ����طŻ�����
                if not self.prioritized:
                    self.buffer.store(state, action, next_state, reward, done)
                else:
                    if len(self.buffer) > self.learning_start_batch:
                        next_state = torch.tensor(next_state, dtype=torch.float, device=self.agent.device)
                        state = torch.tensor(state, dtype=torch.float, device=self.agent.device)
                        with torch.no_grad():
                            # ����TDĿ��
                            td_target = self.agent.calculate_td_target(next_state, reward, float(done))
                            # ����td���
                            td_error = td_target - self.agent.policy_model(state)[action.item()]
                    else:
                        # ���绹û�п�ʼѵ����ʱ�򣬸�����ͬ��td��ʹ�þ��ȳ���
                        td_error = 1.0
                    self.buffer.store(td_error.item(), state, action, next_state, reward, done)

                # ѵ��������
                if len(self.buffer) > self.learning_start_batch * self.buffer.batch_size:
                    per_data = self.agent.optimize(self.buffer.sample())
                    if self.prioritized:
                        index, td_error = per_data
                        self.buffer.update(index, td_error)

                # ����
                state = next_state.copy()

            print(f"�� {i} ��ѵ�����")


class DDPG(OffPolicyModel):
    """
    DDPGģ��
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
        DDPGģ��
        :param env: ����
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: Ŀ�������ϱ���
        :param update_interval: Ŀ��������¼������
        :param grad_norm_clipping: �ݶȲü���Χ
        :param buffer_size: ��������С
        :param batch_size: ����С
        :param learning_start_batch:
        :param selector_sigma_max_val:
        :param selector_sigma_min_val:
        :param selector_sigma_decay_step
        :param device: �����豸
        """
        super(DDPG, self).__init__(env, device)

        # ��ʼѧϰ����Ҫ������
        self.learning_start_batch = learning_start_batch
        # �����Ƿ񱻽���Ϊ����
        self.action_as_index = False

        # ��������
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

        # ��������ѡ����
        selector = utils.selector.GaussianSelector(selector_sigma_max_val,
                                                   selector_sigma_min_val,
                                                   selector_sigma_decay_step,
                                                   model=online_policy_model,
                                                   action_bounds=self.env_info["bounds"])
        # �����طŻ�����
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], self.env_info["action_dim"],
                                                buffer_size, batch_size, prioritized=False,
                                                alpha=None, beta=None)
        # ����������
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
    TD3ģ��
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
        ���캯��
        :param env: �����廷��
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: Ŀ��������ϵ��
        :param update_interval: ���¼������
        :param grad_norm_clipping: �ݶȲü���ֵ
        :param buffer_size: ��������С
        :param batch_size: ����С
        :param learning_start_batch: ���翪ʼѧϰ������С
        :param target_noise_clipping: ���������ü���Χ
        :param noise_sigma: ����������׼��
        :param selector_sigma_max_val: ����ѡ������׼�����ֵ
        :param selector_sigma_min_val: ����ѡ������׼�����ֵ
        :param selector_sigma_decay_step: ����ѡ������׼��˥������
        :param device: �����豸
        """
        super(TD3, self).__init__(env, device)

        # ��ʼѧϰ����Ҫ������
        self.learning_start_batch = learning_start_batch
        # �����Ƿ񱻽���Ϊ����
        self.action_as_index = False

        # ��������
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

        # ��������ѡ����
        selector = utils.selector.GaussianSelector(selector_sigma_max_val,
                                                   selector_sigma_min_val,
                                                   selector_sigma_decay_step,
                                                   model=online_policy_model,
                                                   action_bounds=self.env_info["bounds"])
        # �����طŻ�����
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], self.env_info["action_dim"],
                                                buffer_size, batch_size, prioritized=False,
                                                alpha=None, beta=None)
        # ����������
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
    SACģ��
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
        ���캯��
        :param env: ����
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param target_entropy: Ŀ����
        :param entropy_lr: ��ѧϰ��
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param tau: Ŀ��������±���
        :param update_interval: Ŀ��������²���
        :param grad_norm_clipping: �ݶȲü���Χ
        :param buffer_size: ��������С
        :param batch_size: ����С
        :param learning_start_batch: ��ʼѧϰʱ�Ѽ���������
        :param device: �����豸
        """
        super(SAC, self).__init__(env, device)

        # ��ʼѧϰ����Ҫ������
        self.learning_start_batch = learning_start_batch
        # �����Ƿ񱻽���Ϊ����
        self.action_as_index = False
        # Ŀ����
        if target_entropy is None:
            target_entropy = -np.prod(self.env.action_space.shape).item()

        # ��������
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

        # �����طŻ�����
        self.buffer = build.build_replay_buffer(self.env_info["state_dim"], self.env_info["action_dim"],
                                                buffer_size, batch_size, prioritized=False,
                                                alpha=None, beta=None)
        # ����������
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
    REINFORCEģ��
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
        ���캯��
        :param env: ����
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param trajectory_max_length: �����켣��󳤶�
        :param device: �����豸
        """
        super(REINFORCE, self).__init__(env, device)

        # ��������
        policy_model, _ = self.generate_net(self.env_info,
                                            module_config,
                                            device,
                                            ignore_value=True,
                                            policy_build_handle=policy_build_handle,
                                            value_build_handle=None)
        # �켣������
        self.buffer = utils.buffer.TrajectoryBuffer(self.env_info["state_dim"], 1,
                                                    gamma=gamma, capacity=trajectory_max_length, device=device)
        # ������
        self.agent = sarl.REINFORCEAgent(policy_model,
                                         gamma=gamma,
                                         lr=lr,
                                         grad_norm_clipping=grad_norm_clipping)


class VPG(OnPolicyModel):
    """
    VPGģ��
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
        VPGģ��
        :param env: ����
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param entropy_loss_weight: ����ʧȨ��
        :param trajectory_max_length: �����켣��󳤶�
        :param device: �����豸
        """
        super(VPG, self).__init__(env, device)

        # ��������
        policy_model, value_model = self.generate_net(self.env_info,
                                                      module_config,
                                                      device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)

        # �켣������
        self.buffer = utils.buffer.TrajectoryBuffer(self.env_info["state_dim"], 1,
                                                    gamma=gamma, capacity=trajectory_max_length, device=device)
        # ������
        self.agent = sarl.VPGAgent(policy_model,
                                   value_model,
                                   gamma=gamma,
                                   lr=lr,
                                   grad_norm_clipping=grad_norm_clipping,
                                   entropy_loss_weight=entropy_loss_weight)


class A3C(OnPolicyModel):
    """
    A3Cģ��
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
        ���캯��
        :param env_build_handle: ������������
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param bootstrap_steps: �Ծٲ���
        :param worker_num: ����������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param entropy_loss_weight: ����ʧȨ��
        :param trajectory_max_length: �켣��󳤶�
        """
        super(A3C, self).__init__(env_build_handle(), "cpu")

        # ������������
        self.env_build_handle = env_build_handle
        # �Ծٲ���
        self.bootstrap_steps = bootstrap_steps
        # ����������
        self.worker_num = worker_num
        # �������б�
        self.worker: list[mp.Process] = []

        # ������������
        self.shared_policy_model, self.shared_value_model = self.generate_net(self.env_info,
                                                                              module_config,
                                                                              device='cpu',
                                                                              ignore_value=False,
                                                                              policy_build_handle=policy_build_handle,
                                                                              value_build_handle=value_build_handle)
        # �����豸��ǿ�ƴ���CPU
        self.shared_policy_model.to(self.device)
        self.shared_value_model.to(self.device)
        # �����ڴ�
        self.shared_policy_model = self.shared_policy_model.share_memory()
        self.shared_value_model = self.shared_value_model.share_memory()
        # ���������Ż���
        self.shared_policy_optimizer = utils.share_optim.SharedAdam(self.shared_policy_model.parameters(), lr=lr)
        self.shared_value_optimizer = utils.share_optim.SharedAdam(self.shared_value_model.parameters(), lr=lr)

        # �洢����
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
        A3C������
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
        # ��ȡ��ǰ����
        cur_process = mp.current_process()
        # ��������
        env = env_build_handle()
        # ���ز�������ͼ�ֵ����
        policy_model, value_model = OnPolicyModel.generate_net(OnPolicyModel.get_env_param(env),
                                                               module_config,
                                                               device='cpu',
                                                               ignore_value=False,
                                                               policy_build_handle=policy_build_handle,
                                                               value_build_handle=value_build_handle)
        # �켣������
        buffer = utils.buffer.TrajectoryBuffer(state_dim, 1,
                                               gamma=gamma, capacity=trajectory_max_length, device='cpu')
        # ������
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
            # ���û���
            state, _ = env.reset(seed=env_seed)
            done = False
            # ��ǰ����
            step = 0

            while not done:
                # ѡ����
                action, _, _ = agent.select_action(state, a_type="train")
                # ����
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # �洢
                buffer.store(state, action, float(reward))
                # ����״̬
                state = next_state.copy()

                # �ж��Ƿ񵽴��Ծٲ���
                if done or step % bootstrap_steps == 0:
                    # ��һʱ��״ֵ̬
                    v_next = torch.zeros([0], dtype=torch.float32, device='cpu') if done \
                        else agent.value_model(torch.tensor(state, dtype=torch.float32, device='cpu'))
                    # ����
                    agent.optimize(buffer.recall(v_next))
                    # ��������
                    buffer.clear()
            # ��ʾ
            print(f"���� {cur_process.name}, �� {i} ��ѵ������")

    def learn(self, episode: int, env_seed=None):
        # ���������ĳ�ʼ������
        if isinstance(env_seed, int) or env_seed is None:
            env_seed = [env_seed for _ in range(self.worker_num)]
        # ��ȡ״̬ά��

        # ����̹�����
        self.worker = [mp.Process(target=A3C.a3c_worker, args=(*self.worker_args,
                                                               self.env_info["state_dim"],
                                                               episode,
                                                               env_seed[i]))
                       for i in range(self.worker_num)]
        # ���̿�ʼ
        for worker in self.worker:
            worker.start()
        # �ȴ����̽���
        for worker in self.worker:
            worker.join()


class A2C(OnPolicyModel):
    """
    A2Cģ��
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
        ���캯��
        :param env: ���л���
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param gae_lambda: GAE�Ľ����ۿ�ϵ��
        :param entropy_loss_weight: ����ʧȨ��
        :param device: �����豸
        """
        assert isinstance(env, gymnasium.vector.VectorEnv), f"��Ҫ������ǲ��л���VecEnv��ʵ��Ϊ {type(env)}"

        super(A2C, self).__init__(env, device)

        # ��������
        policy_model, value_model = self.generate_net(self.env_info,
                                                      module_config,
                                                      device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)

        # ������
        self.agent = sarl.A2CAgent(policy_model,
                                   value_model,
                                   gamma=gamma,
                                   lr=lr,
                                   grad_norm_clipping=grad_norm_clipping,
                                   gae_lambda=gae_lambda,
                                   entropy_loss_weight=entropy_loss_weight,
                                   gae_norm=gae_norm)

        # �켣������
        self.buffer = utils.buffer.RolloutBuffer(self.env_info["state_dim"], 1,
                                                 self.env,
                                                 buffer_size, None, gamma, gae_lambda, device)

    def learn(self, episode: int, env_seed=None):
        """
        ģ��ѧϰ
        :param episode: ѵ���ִ�
        :param env_seed: ��������
        :return:
        """
        for i in range(episode):
            # ѵ��
            self.agent.optimize(self.buffer, seed=env_seed)
            # ��ӡѵ�����
            print(f"�� {i} ��ѵ������")


class PPO(OnPolicyModel):
    """
    A2Cģ��
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
        ���캯��
        :param env: ���л���
        :param module_config: ����������
        :param policy_build_handle: �������紴������
        :param value_build_handle: ��ֵ���紴������
        :param gamma: �ۿ�ϵ��
        :param lr: ѧϰ��
        :param grad_norm_clipping: �ݶȲü���Χ
        :param gae_lambda: GAE�Ľ����ۿ�ϵ��
        :param gae_norm: �Ƿ��GAE���б�׼��
        :param entropy_loss_weight: ����ʧȨ��
        :param buffer_size: �켣��������С
        :param batch_size: ��������С
        :param update_epoch: �����ظ����´���
        :param policy_clipping: �����¾ɸ��ʱȲü���Χ
        :param value_clipping: �¾ɼ�ֵ֮��ü���Χ
        :param device: �����豸
        """
        assert isinstance(env, gymnasium.vector.VectorEnv), f"��Ҫ������ǲ��л���VecEnv��ʵ��Ϊ {type(env)}"

        super(PPO, self).__init__(env, device)

        # ��ȡ״̬����ά��


        # ��������
        policy_model, value_model = self.generate_net(self.env_info,
                                                      module_config,
                                                      device,
                                                      ignore_value=False,
                                                      policy_build_handle=policy_build_handle,
                                                      value_build_handle=value_build_handle)

        # ����������
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

        # �켣������
        self.buffer = utils.buffer.RolloutBuffer(self.env_info["state_dim"], 1,
                                                 self.env,
                                                 buffer_size, batch_size, gamma, gae_lambda, device)

    def learn(self, episode: int, env_seed=None):
        """
        ģ��ѧϰ
        :param episode: ѵ���ִ�
        :param env_seed: ��������
        :return:
        """
        for i in range(episode):
            # ѵ��
            self.agent.optimize(self.buffer, seed=env_seed)
            # ��ӡѵ�����
            print(f"�� {i} ��ѵ������")

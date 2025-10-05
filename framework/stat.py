# -*- coding:gbk -*-

"""
ѵ�����ݺ���
"""


import gymnasium
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["show_episode_statistics"]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_reward(reward, window: int = 50):
    """
    ƽ����������
    :param reward: ����ֵ�б�
    :param window: ���ڳ���
    :return:
    """
    return np.convolve(reward, np.ones(window) / window, 'same')


def draw_reward(reward: np.ndarray, window: int = 50, filename: str = None, **plot_kwargs):
    """
    ���ƽ���
    :param reward: ����ֵ�б�
    :param window: ���ڳ���
    :param filename: ����·��
    :return:
    """
    plt.figure()
    # ����ԭʼ����
    plt.plot(reward, linewidth=3, alpha=0.5)
    # ����ƽ����Ľ���
    plt.plot(smooth_reward(reward, window), linewidth=2, **plot_kwargs)
    # ���ಿ��
    plt.legend(["ԭʼ����", "ƽ������"])
    plt.xlabel("ѵ���ִ�")
    plt.ylabel("����ֵ")
    # ����
    if filename is not None:
        plt.savefig(filename)
    # չʾ
    plt.show()


def draw_length(length: np.ndarray, filename: str = None, **plot_kwargs):
    """
    ����ÿ�ֵĽ�������
    :param length: ���������б�
    :param filename: ����·��
    :return:
    """
    plt.figure()
    # ����ԭʼ����
    plt.plot(length, linewidth=2, **plot_kwargs)
    # ���ಿ��
    plt.xlabel("ѵ���ִ�")
    plt.ylabel("��������")
    # ����
    if filename is not None:
        plt.savefig(filename)
    # չʾ
    plt.show()


def draw_time(times: np.ndarray, filename: str = None, **plot_kwargs):
    """
    �������һ�ֽ�����ʱ��
    :param times: ����ʱ���б�
    :param filename: ����·��
    :return:
    """
    plt.figure()
    # ����ԭʼ����
    plt.plot(times, linewidth=2, **plot_kwargs)
    # ���ಿ��
    plt.xlabel("ѵ���ִ�")
    plt.ylabel("����ʱ��")
    # ����
    if filename is not None:
        plt.savefig(filename)
    # չʾ
    plt.show()


def show_episode_statistics(monitor: gymnasium.wrappers.RecordEpisodeStatistics,
                            window: int = 10,
                            length_arr_file: str = "length.npy",
                            returns_arr_file: str = "returns.npy",
                            time_arr_file: str = "times.npy",
                            length_plt_file: str = "length.png",
                            returns_plt_file: str = 'returns.png',
                            time_plt_file: str = 'times.png',):
    """
    ��ʾ������ѵ���ִ���Ϣ
    :param monitor:
    :param window:
    :param length_arr_file:
    :param returns_arr_file:
    :param time_arr_file:
    :param length_plt_file:
    :param returns_plt_file:
    :param time_plt_file:
    :return:
    """
    episode_lengths = np.array(monitor.length_queue)
    episode_returns = np.array(monitor.return_queue)
    episode_time = np.array(monitor.time_queue)

    # ��������
    if length_arr_file is not None:
        np.save(length_arr_file, episode_lengths)
    if returns_arr_file is not None:
        np.save(returns_arr_file, episode_returns)
    if time_arr_file is not None:
        np.save(time_arr_file, episode_time)

    # ����ͼ��
    draw_reward(episode_returns, window, returns_plt_file)
    draw_length(episode_lengths, length_plt_file)
    draw_time(episode_time, time_plt_file)

    return episode_lengths, episode_returns, episode_time

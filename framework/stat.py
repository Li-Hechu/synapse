# -*- coding:gbk -*-

"""
训练数据后处理
"""


import gymnasium
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["show_episode_statistics"]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_reward(reward, window: int = 50):
    """
    平滑奖励函数
    :param reward: 奖励值列表
    :param window: 窗口长度
    :return:
    """
    return np.convolve(reward, np.ones(window) / window, 'same')


def draw_reward(reward: np.ndarray, window: int = 50, filename: str = None, **plot_kwargs):
    """
    绘制奖励
    :param reward: 奖励值列表
    :param window: 窗口长度
    :param filename: 保存路径
    :return:
    """
    plt.figure()
    # 画出原始奖励
    plt.plot(reward, linewidth=3, alpha=0.5)
    # 画出平滑后的奖励
    plt.plot(smooth_reward(reward, window), linewidth=2, **plot_kwargs)
    # 其余部件
    plt.legend(["原始奖励", "平滑后奖励"])
    plt.xlabel("训练轮次")
    plt.ylabel("奖励值")
    # 保存
    if filename is not None:
        plt.savefig(filename)
    # 展示
    plt.show()


def draw_length(length: np.ndarray, filename: str = None, **plot_kwargs):
    """
    绘制每轮的交互步数
    :param length: 交互步数列表
    :param filename: 保存路径
    :return:
    """
    plt.figure()
    # 画出原始数据
    plt.plot(length, linewidth=2, **plot_kwargs)
    # 其余部件
    plt.xlabel("训练轮次")
    plt.ylabel("交互步数")
    # 保存
    if filename is not None:
        plt.savefig(filename)
    # 展示
    plt.show()


def draw_time(times: np.ndarray, filename: str = None, **plot_kwargs):
    """
    绘制完成一轮交互的时间
    :param times: 交互时间列表
    :param filename: 保存路径
    :return:
    """
    plt.figure()
    # 画出原始数据
    plt.plot(times, linewidth=2, **plot_kwargs)
    # 其余部件
    plt.xlabel("训练轮次")
    plt.ylabel("交互时长")
    # 保存
    if filename is not None:
        plt.savefig(filename)
    # 展示
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
    显示并保存训练轮次信息
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

    # 保存数据
    if length_arr_file is not None:
        np.save(length_arr_file, episode_lengths)
    if returns_arr_file is not None:
        np.save(returns_arr_file, episode_returns)
    if time_arr_file is not None:
        np.save(time_arr_file, episode_time)

    # 绘制图像
    draw_reward(episode_returns, window, returns_plt_file)
    draw_length(episode_lengths, length_plt_file)
    draw_time(episode_time, time_plt_file)

    return episode_lengths, episode_returns, episode_time

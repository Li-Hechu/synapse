# -*- coding:gbk -*-

import numpy as np


class SumTree:
    """
    求和二叉树
    """
    def __init__(self, capacity):
        """
        构造函数
        :param capacity: 节点个数
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # 二叉树数组结构
        self.data = np.zeros(capacity, dtype=object)  # 存储经验
        self.write = 0  # 写入指针

    def add(self, priority, data):
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # 向上更新
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):  # 到达叶子
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    v -= self.tree[left_child]
                    parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        # 根节点存储总和
        return self.tree[0]

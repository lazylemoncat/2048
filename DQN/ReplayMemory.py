import itertools
import os
import numpy as np
from collections import deque
import pickle
import random

# 优先级经验回放
class ReplayMemory:
  def __init__(self, capacity, alpha=0.6):
    self.memory = deque(maxlen=capacity)  # 经验回放缓存队列
    self.alpha = alpha  # 优先级指数
    self.priorities = np.zeros((capacity,), dtype=np.float32)  # 优先级数组
    self.max_priority = 1.0  # 最大优先级

  def isFull(self):
    return len(self.memory) == self.memory.maxlen  # 判断经验回放缓存是否已满

  def push(self, transition):
    # 如果缓存已满，则移除最旧的经验
    if self.isFull():
      self.memory.popleft()  # 使用popleft移除最左边的元素，即最旧的经验
      self.priorities = np.roll(self.priorities, -1)  # 将priorities数组向左滚动一个位置
      self.priorities[-1] = 0  # 将新的位置的优先级设置为0
    else:
      self.priorities = np.roll(self.priorities, 1)  # 将priorities数组向右滚动一个位置
      self.priorities[0] = 0  # 将新的位置的优先级设置为0

    self.memory.append(transition)  # 将新的经验添加到缓存队列
    idx = self._next_idx() - 1  # 获取新经验的索引
    self.priorities[idx] = self.max_priority  # 设置新经验的优先级为最大优先级

  def _next_idx(self):
    return (self.memory.index(None) if None in self.memory else len(self.memory))  # 获取下一个可用的索引

  def sample(self, batch_size):
    assert batch_size <= len(self.memory), "Batch size is larger than memory size!"
    batch = []
    idxes = []
    # 使用itertools.islice来创建一个迭代器，它返回有效的经验回放段
    valid_segment = itertools.islice(self.memory, 0, self._next_idx())
     # 获取经验回放段中的最小优先级，并确保它不是0
    p_min = max(np.min(self.priorities[:self._next_idx()]), 1e-10)
    max_weight = p_min ** (-self.alpha)
    for idx, transition in enumerate(valid_segment):
      prob = self.priorities[idx] ** self.alpha
      weight = prob / max_weight
      idxes.append(idx)
      batch.append((idx, weight))
    batch = random.sample(batch, batch_size)
    idxes = [x[0] for x in batch]
    weights = [x[1] for x in batch]
    return idxes, weights

  def update_priority(self, idx, priority):
    self.priorities[idx] = priority  # 更新指定索引的优先级

  def _update_priority(self, idx, priority):
    old_priority = self.priorities[idx]
    self.priorities[idx] = priority  # 更新指定索引的优先级
    if priority > old_priority:
        self.max_priority = max(self.max_priority, priority)  # 更新最大优先级

  def __len__(self):
    return len(self.memory)  # 返回经验回放缓存的大小

  def save_memory(self, filename='memory.pkl'):
    memory_data = [self.memory, self.priorities]  # 保存经验回放缓存和优先级数组
    with open(filename, 'wb') as f:
        pickle.dump(memory_data, f)  # 将数据保存到文件

  def load_memory(self, filename='memory.pkl'):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            memory_data = pickle.load(f)  # 从文件加载数据
            self.memory, self.priorities = memory_data  # 更新经验回放缓存和优先级数组

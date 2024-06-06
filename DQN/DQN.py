'''
算法使用的DQN,通过计算Q值选择动作
DDQN,使用两个网络,一个用于选择动作,一个用于计算Q值
经验回放，用于存储经验，随机抽取经验进行学习
采用探索和贪心的策略,一定概率随机选择动作,一定概率选择Q值最大的动作
将模型和数据搭载到GPU上进行训练,提高训练速度
'''

import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from DQN.Net import Net
from DQN.ReplayMemory import ReplayMemory

# 定义DQN
class DQN(torch.nn.Module):
  def __init__(
      self, 
      # 动作集
      actions,
      # 输入数组长度 
      input_size, 
      # 输出数组长度
      output_size, 
      # 记忆库大小
      memory_capacity=10000,
      # 学习率
      lr=None, 
      # 模型保存路径
      path=None,
      # 设备
      device='cpu'
    ):
    super(DQN, self).__init__()
    # 开始训练时间
    self.start_time = time.time()
    # 已训练时间
    self.trained_time = 0
    # 动作空间
    self.actions = actions
    # 用于预测当前状态下最佳动作的Q值。
    self.policy_net = Net(input_size, output_size)
    # 参数定期从策略网络复制过来。
    self.target_net = Net(input_size, output_size)
    # 目标网络的参数与策略网络相同
    self.target_net.load_state_dict(self.policy_net.state_dict())
    # 创建一个大小为memory_capacity的经验回放记忆库，用于存储代理的经验
    self.memory = ReplayMemory(memory_capacity)
    # 创建一个Adam优化器，用于更新策略网络的参数
    if lr is None:
      self.optimizer = optim.Adam(self.policy_net.parameters())
    else:
      self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
    # 定义一个均方误差损失函数，用于DQN的学习过程
    self.criterion = nn.MSELoss()
    # 如果有已经训练好的模型，加载模型
    self.path = path
    if path is not None:
      self.load_model(path)
    else:
      self.path = "model.pth"
      self.load_model()
    # 设备
    self.device = device
    self.last_action = None
    

  def select_action(self, state, epsilon=0.1):
    if random.random() < epsilon:
      return self.actions[random.randint(0, len(self.actions)-1)]
    # 转化为1*n的张量
    state = torch.tensor(state, dtype=torch.float).view(1, -1)
    # 不计算梯度
    with torch.no_grad():
      # 选择Q值最大的动作
      state = state.to(self.device)
      arr = self.policy_net(state)
      sorted_indices = torch.argsort(arr, descending=True)
      action = self.actions[sorted_indices[0][0].item()]
      return action
      if action != self.last_action:
        self.last_action = action
        return action
      else:
        action = self.actions[sorted_indices[0][1].item()]
        self.last_action = action
        return action
    
  def update_priority(self, idxes, current_q_values, expected_q_values):
    # 更新经验的优先级
    with torch.no_grad():
      td_errors = torch.abs(current_q_values - expected_q_values.unsqueeze(1))
      for i, td_error in enumerate(td_errors):
        self.memory.update_priority(idxes[i], td_error.item())
  
  def learn(self, batch_size, discount=0.9):
    if len(self.memory) < batch_size:
      return
    # 从经验回放缓冲区中随机抽取批量大小的经验，并获取对应的权重
    idxes, weights = self.memory.sample(batch_size)
    transitions = [self.memory.memory[i] for i in idxes]
    # 解包这些经验，得到状态、动作、下一个状态和奖励
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    # 将动作转换为索引
    batch_action = [self.actions.index(a) for a in batch_action]
    # 将状态、动作、下一个状态和奖励转换为PyTorch张量
    batch_state = np.array(batch_state)
    batch_state = torch.tensor(batch_state, dtype=torch.float)
    batch_action = torch.tensor(batch_action, dtype=torch.long)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float)
    batch_next_state = np.array(batch_next_state)
    batch_next_state = torch.tensor(batch_next_state, dtype=torch.float)
    # 设备a
    batch_state = batch_state.to(self.device)
    batch_action = batch_action.to(self.device)
    batch_reward = batch_reward.to(self.device)
    batch_next_state = batch_next_state.to(self.device)
    # 使用当前策略网络计算当前状态下的Q值
    # `gather(1, batch_action)` 获取每个状态-动作对的Q值
    batch_action = batch_action.unsqueeze(1)
    current_q_values = self.policy_net(batch_state).gather(1, batch_action)
    # 使用目标网络计算下一个状态的最大Q值
    # `max(1)[0].detach()` 获取每个下一个状态的最大Q值，并detach以防止梯度回传
    max_next_q_values = self.target_net(batch_next_state).max(1)[0].detach()
    # 计算期望Q值，即奖励加上下一个状态的最大Q值乘以折扣因子
    expected_q_values = batch_reward + discount * max_next_q_values
    # 使用均方误差损失函数计算损失
    loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
    # 清空梯度
    self.optimizer.zero_grad()
    # 反向传播计算损失
    loss.backward()
    # 更新模型参数
    self.optimizer.step()
    # 更新经验的优先级
    self.update_priority(idxes, current_q_values, expected_q_values)

  def update(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def save_model(self, filename='model.pth'):
    if self.path is None:
      self.path = filename
    torch.save({
      'state_dict': self.policy_net.state_dict(),
      'train_time': int(time.time() - self.start_time + self.trained_time),
    }, filename)

  def load_model(self, filename='model.pth'):
    if self.path is not None:
      filename = self.path
    if os.path.isfile(filename):
      DQN_CNN_onehot = Net
      model_params = torch.load(filename)
      self.policy_net.load_state_dict(model_params['state_dict'])
      self.target_net.load_state_dict(model_params['state_dict'])
      if 'train_time' in model_params:
        self.trained_time = model_params['train_time']
  
  def get_train_time(self):
    if self.path is not None:
      filename = self.path
      if os.path.isfile(filename):
        model_params = torch.load(filename)
        if 'train_time' in model_params:
          return model_params['train_time']
        else:
          return 0
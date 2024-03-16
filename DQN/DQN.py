import torch
import torch.nn as nn
import torch.optim as optim
import os
from DQN.Net import Net
from DQN.ReplayMemory import ReplayMemory

# 定义DQN
class DQN:
  def __init__(self):
    # 用于预测当前状态下最佳动作的Q值。
    self.policy_net = Net()
    # 参数定期从策略网络复制过来。
    self.target_net = Net()
    # 目标网络的参数与策略网络相同
    self.target_net.load_state_dict(self.policy_net.state_dict())
    # 创建一个大小为10000的经验回放记忆库，用于存储代理的经验
    self.memory = ReplayMemory(10000)
    # 创建一个Adam优化器，用于更新策略网络的参数
    self.optimizer = optim.Adam(self.policy_net.parameters())
    # 定义一个均方误差损失函数，用于DQN的学习过程
    self.criterion = nn.MSELoss()
    # 如果有已经训练好的模型，可以加载模型
    self.load_model()

  def select_action(self, state):
    action = ['w', 'a', 's', 'd']
    state = torch.tensor(state, dtype=torch.float).view(1, -1)
    with torch.no_grad():
      index = self.policy_net(state).max(1)[1].view(1, 1).item()
      return action[index]
  
  def learn(self, batch_size, discount=0.99):
    if len(self.memory) < batch_size:
      return
    # 从经验回放缓冲区中随机抽取批量大小的经验
    transitions = self.memory.sample(batch_size)
    # 解包这些经验，得到状态、动作、下一个状态和奖励
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    actions = ['w', 'a', 's', 'd']
    batch_action = [actions.index(a) for a in batch_action]
    # 将状态、动作、下一个状态和奖励转换为PyTorch张量
    batch_state = torch.tensor(batch_state, dtype=torch.float)
    batch_action = torch.tensor(batch_action, dtype=torch.long)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float)
    batch_next_state = torch.tensor(batch_next_state, dtype=torch.float)
    # 使用当前策略网络计算当前状态下的Q值
    # `gather(1, batch_action)` 获取每个状态-动作对的Q值
    batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1)
    # print(self.policy_net(batch_state)[0], batch_action)
    current_q_values = self.policy_net(batch_state)[0].gather(1, batch_action)
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

  def update(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def save_model(self, filename='model.pth'):
    torch.save(self.policy_net.state_dict(), filename)

  def load_model(self, filename='model.pth'):
    if os.path.isfile(filename):
      self.policy_net.load_state_dict(torch.load(filename))
      self.target_net.load_state_dict(torch.load(filename))
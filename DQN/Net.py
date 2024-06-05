'''
激活函数使用LeakyReLU,避免输入为0或负数时神经元死亡
输入为一维数组表示的状态
输出为一维数组表示的动作的Q值
dueling dqn对偶网络
v网络计算出状态的价值
a网络计算出每个动作的优劣
最终输出为v+a-a.mean()
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络模型
class Net(nn.Module):
  # # 输入一个1*input_size的张量，输出一个1*output_size的张量
  # def __init__(self, input_size, output_size, neurons=32):
  #   super(Net, self).__init__()
  #   self.fc1 = nn.Linear(input_size, neurons)
  #   self.fcs = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(4)])
  #   self.vs = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(4)])
  #   self.v5 = nn.Linear(neurons, 1)
  #   self.acs = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(4)])
  #   self.a5 = nn.Linear(neurons, output_size)

  # def forward(self, x):
  #   sigmoid = nn.Sigmoid()
  #   x = sigmoid(self.fc1(x))
  #   x = torch.nn.functional.normalize(x, dim=1)
  #   for fc in self.fcs:
  #     x = sigmoid(fc(x))
  #     x = torch.nn.functional.normalize(x, dim=1)
  #   for v in self.vs:
  #     v = sigmoid(v(x))
  #     v = torch.nn.functional.normalize(v, dim=1)
  #   v = self.v5(v)
  #   for ac in self.acs:
  #     ac = sigmoid(ac(x))
  #     ac = torch.nn.functional.normalize(ac, dim=1)
  #   a = self.a5(ac)
  #   x = v + a - a.mean()
  #   return x
  
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(16,64,(2,2))
    self.conv2 = nn.Conv2d(64,128,(2,2))
    self.conv3 = nn.Conv2d(128,32,(2,2))
    self.fc1 = nn.Linear(32,4)

  def forward(self, x):
    x = x.view(-1,16,4,4).float()
    x = F.relu(self.conv1(x))  # -1,64,3,3
    x = F.relu(self.conv2(x))  # -1,128,2,2
    x = F.relu(self.conv3(x))  # -1,32,1,1
    x= x.view(-1,32)
    y = self.fc1(x)
    return y
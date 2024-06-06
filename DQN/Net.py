'''
激活函数使用LeakyReLU,避免输入为0或负数时神经元死亡
输入为一维数组表示的状态
输出为一维数组表示的动作的Q值
dueling dqn对偶网络
v网络计算出状态的价值
a网络计算出每个动作的优劣
最终输出为v+a-a.mean()
'''

import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络模型
class Net(nn.Module):
  def __init__(self, input_size, output_size):
    super(Net, self).__init__()
    # 定义卷积层
    # 输入通道数为input_size,输出通道数为64,卷积核大小为2*2
    self.conv1 = nn.Conv2d(input_size,64,(2,2))
    self.conv2 = nn.Conv2d(64,128,(2,2))
    self.conv3 = nn.Conv2d(128,32,(2,2))
    # 定义全连接层
    # 输入节点数为32,输出节点数为output_size
    self.fc1 = nn.Linear(32,output_size)

  def forward(self, x):
    # 输入x的维度为-1,16,4,4
    x = x.view(-1,16,4,4).float()
    x = F.relu(self.conv1(x))  # -1,64,3,3
    x = F.relu(self.conv2(x))  # -1,128,2,2
    x = F.relu(self.conv3(x))  # -1,32,1,1
    x= x.view(-1,32)
    y = self.fc1(x)
    return y
import torch
import torch.nn as nn

# 定义神经网络模型
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(16, 64)
    self.fc2 = nn.Linear(64, 4)

  def forward(self, x):
    # x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    return self.fc2(x)
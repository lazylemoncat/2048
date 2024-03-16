import torch
import torch.nn as nn
import torch.optim as optim
import random
from Game import Game
import threading
from DQN.Typer import Typer
from DQN.DQN import DQN

def choseAction(actions):
  pass
  return random.choice(actions)

def calculateQValue(model, state, actions, action, learning_rate=0.001):
  q_values = model(state)
  q_value = q_values.clone()
  for i, action in enumerate(actions):
    if action == action:
      q_value[0][i] = 1  # 假设当前动作的Q值为1
  return q_value

def runModel(game):
  learning_rate = 0.001
  # 创建DQN模型、损失函数和优化器
  model = DQN()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  typer = Typer()
  actions = ['w', 'a', 's', 'd']
  while game.getGameState() == "not over":
    state = torch.tensor(game.getMatrix(), dtype=torch.float32).view(1, -1)
    # 选择动作
    action = choseAction(actions)
    # 计算Q值
    q_value = calculateQValue(model, state, actions, action, learning_rate)
    # 计算损失
    loss = criterion(q_value, torch.tensor([1.0]))  # 假设我们只对一个动作感兴趣
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"State: \n{state}\nAction: {action}\nQ-Value: \n{q_value}")
    typer.type(action)
  # 保存模型
  torch.save(model.state_dict(), "model.pth")

def main():
  game = Game()
  thread_b = threading.Thread(target=runModel, args=(game,))
  thread_b.start()
  game.run()

if __name__ == "__main__":
  main()
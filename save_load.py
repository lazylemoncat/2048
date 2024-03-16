import pickle
import torch

# 保存模型
def save_model(dqn, filename='model.pth'):
    torch.save(dqn.policy_net.state_dict(), filename)

# 加载模型
def load_model(dqn, filename='model.pth'):
    dqn.policy_net.load_state_dict(torch.load(filename))
    dqn.target_net.load_state_dict(torch.load(filename))
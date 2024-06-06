import os
import threading
import numpy as np
import torch
import tqdm
from env import env as environment
from DQN.DQN import DQN
from verify import verify
from utils.TrainingLogger import TrainingLogger

def one_hot(state):
  state = np.array(state).reshape(4, 4)
  one_hot_state = np.zeros((16, state.shape[0], state.shape[1]), dtype=np.float16)
  basecode = np.eye(16)
  for m in range(state.shape[0]):
      for n in range(state.shape[1]):
          value = state[m, n]
          one_hot_state[:, m, n] = basecode[int(np.log2(value) if value != 0 else 0), :]
  return one_hot_state

def runModel(env, dqn, logger, batch_size=32, episodes=100, is_train=True):
  for _ in tqdm.trange(episodes):
    done = False
    env.reset()
    steps = 0
    while not done:
      observation = env.state
      observation = one_hot(observation)
      action = dqn.select_action(observation, epsilon=0)
      new_observation, reward, done, info = env.step(action)
      new_observation = one_hot(new_observation)
      if reward < 0:
        steps += 1
      else:
        steps = 0
      if steps > 10 and reward < 0:
        dqn.memory.push((observation, action, new_observation, -100))
        break
      dqn.memory.push((observation, action, new_observation, reward))
      observation = new_observation
      if len(dqn.memory) > batch_size and is_train:
        dqn.learn(batch_size)
      if done:
        break

def train(env, dqn):
  i = 0
  logger = TrainingLogger()
  loggerIllegalMove = TrainingLogger()
  while True:
    print("第{}次训练".format(i))
    i += 1
    runModel(env, dqn, logger)
    env.reset()
    average_num, highest_num, IllegalMove = verify(env, dqn, EPISODES=1)
    logger.log(highest_num)
    loggerIllegalMove.log(IllegalMove)
    if i % 10 == 0:
      dqn.save_model(f"models/model2{i}.pth")
      logger.plot()
      loggerIllegalMove.plot()
      logger.clear()
      loggerIllegalMove.clear()
    dqn.update()

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  env = environment(seed=1)
  dqn = DQN(
    actions=env.action_space, 
    input_size=16, 
    output_size=4,
    memory_capacity=10000,
    path="model2.pth",
    device=device
  ).to(device)
  print(f"{dqn.get_train_time()}")
  thread_train = threading.Thread(target=train, args=(env, dqn))
  thread_train.start()
  # env.render()

if __name__ == "__main__":
  main()
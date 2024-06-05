import threading
from time import sleep
import torch
import tqdm
from env import env as environment
from DQN.DQN import DQN
from verify import verify

def runModel(env, dqn, batch_size=32, episodes=100, is_train=True):
  for _ in tqdm.trange(episodes):
    done = False
    env.reset()
    steps = 0
    while not done:
      observation = env.state
      action = dqn.select_action(observation, epsilon=0)
      new_observation, reward, done, info = env.step(action)
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
  while True:
    print("第{}次训练".format(i))
    i += 1
    runModel(env, dqn)
    env.reset()
    average_num = verify(env, dqn, EPISODES=1)
    # if average_num > -1:
    #   break
    dqn.update()
    dqn.save_model()

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  env = environment(seed=1)
  dqn = DQN(
    actions=env.action_space, 
    input_size=16, 
    output_size=4,
    memory_capacity=10000,
    path="model3.pth",
    device=device
  ).to(device)
  print(f"{dqn.get_train_time()}")
  thread_train = threading.Thread(target=train, args=(env, dqn))
  thread_train.start()
  # env.render()

if __name__ == "__main__":
  main()
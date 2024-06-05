import tqdm


def verify(env, model, EPISODES):
  highest_num = 0
  nums = 0

  for episode in tqdm.trange(EPISODES):
    env.reset()
    done = False
    observation = env.state
    steps = 0
    max_num = 0
    while not done:
      action = model.select_action(observation, epsilon=0)
      observation, reward, done, info = env.step(action)
      if reward < 0:
        steps += 1
      if steps > 10 and reward < 0:
        max_num = -1
        break
    if max_num != -1:
      max_num = env.findMax(env.flatten_matrix())
    nums += max_num
    if env.findMax(env.flatten_matrix()) > highest_num:
      highest_num = env.findMax(env.flatten_matrix())
  average_num = nums / EPISODES
  print(f"Average max num: {average_num},highest num:{highest_num}")
  return average_num

if __name__ == "__main__":
  from env import env as environment
  from DQN.DQN import DQN
  import torch
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
  verify(env, dqn, EPISODES=1)
  env.render()
import math
import random
from time import sleep
import gymnasium as gym
from utils.GameState import GameState
from utils.KeyListener import KeyListener
from utils.Matrix import Matrix
from utils.Move import Move
from utils.Window import Window

class IllegalMove(Exception):
  pass

class env(gym.Env):
  def __init__(self, seed=None):
      random.seed(seed)
      # 初始化游戏状态
      self.gameState = GameState()
      # 动作集
      self.action_space = ['w', 'd', 's', 'a']
      # 初始化矩阵
      self.matrix = []
      # 游戏窗口
      self.window = None
      # 初始化游戏
      self.reset()
      # 游戏状态为将矩阵转化为一维数组
      self.state = self.flatten_matrix()
      # 分数
      self.score = 0
      # 游戏是否结束
      self.__done = False

  def add_two(self, matrix):
      a = random.randint(0, len(matrix)-1)
      b = random.randint(0, len(matrix)-1)
      while matrix[a][b] != 0:
          a = random.randint(0, len(matrix)-1)
          b = random.randint(0, len(matrix)-1)
      matrix[a][b] = 2
      return matrix

  def reset(self):
    self.matrix = []
    for i in range(4):
      self.matrix.append([0] * 4)
    self.matrix = self.add_two(self.matrix)
    self.matrix = self.add_two(self.matrix)
    if self.window is not None:
      self.window.update_grid_cells(self.matrix)
    self.__done = False

  def flatten_matrix(self):
    arr = []
    for row in range(4):
      for col in range(4):
        if (self.matrix[row][col] == 0):
          arr.append(0)
        else:
          arr.append(self.matrix[row][col])
    return arr

  def get_matrix(self):
    arr = []
    for row in range(4):
      for col in range(4):
        arr.append(math.log2(self.matrix[row][col]))
    return arr

  def move(self, action):
      if action == "w":
          self.matrix, isMerged, score = Move.up(self.matrix)
      elif action == "s":
          self.matrix, isMerged, score = Move.down(self.matrix)
      elif action == "a":
          self.matrix, isMerged, score = Move.left(self.matrix)
      elif action == "d":
          self.matrix, isMerged, score = Move.right(self.matrix)
      if self.gameState.game_state(self.matrix) != 'not over':
          self.__done = True
      if isMerged and not self.__done:
          self.matrix = self.add_two(self.matrix)
      return score, isMerged

  def findMax(self, matrix):
      max_val = 0
      for i in range(16):
          if matrix[i] > max_val:
              max_val = matrix[i]
      return max_val

  def are_arrays_equal(self, arr1, arr2):
      # 检查维度是否相同
      if len(arr1) != len(arr2):
          return False
      for i in range(len(arr1)):
          if arr1[i] != arr2[i]:
              return False
      return True

  def getReward(self, score, isMerged):
    self.score += score
    reward = float(score)
    if not isMerged:
      reward = -10
    return reward

  def step(self, action):
      old_state = self.flatten_matrix()
      score, isMerged = self.move(action)
      self.state = self.flatten_matrix()
      reward = self.getReward(score, isMerged)
      info = {
          "score": score,
          "old_state": old_state,
          "action": action,
          "reward": reward,
          "isMerged": isMerged,
      }
      if self.window is not None:
          self.window.update_grid_cells(self.matrix)
      return self.state, reward, self.__done, info

  # 键盘事件
  def __key_down(self, event):
      self.matrix, done = KeyListener.key_down(event, self.matrix)
      if done:
          self.matrix = Matrix.add_two(self.matrix)
          self.window.update_grid_cells(self.matrix)
          grid_cells = self.window.getGridCells()
          self.gameState.showState(grid_cells, self.matrix)
      self.__done = done

  def render(self):
      self.window = Window(self.__key_down)
      # 刷新格子文本组件
      self.window.update_grid_cells(self.matrix)
      self.window.mainloop()

  def close(self):
      pass

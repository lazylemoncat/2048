from utils.Move import Move
from utils.Window import Window
from utils.Matrix import Matrix
from utils.GameState import GameState
from utils.KeyListener import KeyListener
from constants.size import *

# 继承Frame类
#todo steps
#todo scores
class Game():
  def __init__(self):
    # 初始化窗口
    self.Window = Window(self.__key_down)
    # 初始化游戏状态
    self.gameState = GameState()
    # 初始化矩阵
    self.matrix = Matrix.new_game(GRID_LEN)
    # 刷新格子文本组件
    self.Window.update_grid_cells(self.matrix)
    # 是否发生了变化
    self.__done = False

  def move(self, direction):
    if direction == "w":
      self.matrix, done = Move.up(self.matrix)
    elif direction == "s":
      self.matrix, done = Move.down(self.matrix)
    elif direction == "a":
      self.matrix, done = Move.left(self.matrix)
    elif direction == "d":
      self.matrix, done = Move.right(self.matrix)
    if done:
      self.matrix = Matrix.add_two(self.matrix)
      self.Window.update_grid_cells(self.matrix)
      grid_cells = self.Window.getGridCells()
      self.gameState.showState(grid_cells, self.matrix)
    self.__done = done

  def run(self):
    self.Window.mainloop()

  def isDone(self):
    return self.__done

  def getGameState(self):
    return self.gameState.getstate()
  
  # 获取矩阵
  def getMatrix(self):
    return self.matrix
  
  # 键盘事件
  def __key_down(self, event):
    self.matrix, done = KeyListener.key_down(event, self.matrix)
    if done:
      self.matrix = Matrix.add_two(self.matrix)
      self.Window.update_grid_cells(self.matrix)
      grid_cells = self.Window.getGridCells()
      self.gameState.showState(grid_cells, self.matrix)
    self.__done = done

  def reset(self):
    self.state = "not over"
    self.matrix = Matrix.new_game(GRID_LEN)
    self.Window.update_grid_cells(self.matrix)

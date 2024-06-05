from constants.color import *

class GameState:
  def __init__(self):
    self.__state = "not over"
  
  def getstate(self):
    return self.__state

  # 检查矩阵中是否有0
  def __check_empty(matrix):
    for i in range(len(matrix)):
      for j in range(len(matrix[0])):
        if matrix[i][j] == 0:
          return True
    return False

  # 检查相邻的单元格是否相同
  def __check_same_adjacent(matrix):
    for i in range(len(matrix)-1):
      for j in range(len(matrix[0])-1):
        if matrix[i][j] == matrix[i+1][j] or matrix[i][j+1] == matrix[i][j]:
          return True
      for k in range(len(matrix)-1):
        if matrix[len(matrix)-1][k] == matrix[len(matrix)-1][k+1]:
          return True
      for j in range(len(matrix)-1):
        if matrix[j][len(matrix)-1] == matrix[j+1][len(matrix)-1]:
          return True
    return False

  # 检查游戏状态
  def game_state(self, matrix):
    flag = False
    if GameState.__check_empty(matrix):
      flag = True
      self.__state = 'not over'
    if GameState.__check_same_adjacent(matrix):
      flag = True
      self.__state = 'not over'
    elif not flag:
      self.__state = 'lose'
    return self.__state
  
  def showState(self, grid_cells, matrix):
    if self.game_state(matrix) == 'not over':
      return
    if self.__state == 'lose':
      grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
      grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)
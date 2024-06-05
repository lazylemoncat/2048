import random
from constants.size import *

class Matrix:
  # 创建一个n行n列的矩阵，每个元素初始化为0
  def new_game(n):
    matrix = []
    for i in range(n):
      matrix.append([0] * n)
    matrix = Matrix.add_two(matrix)
    matrix = Matrix.add_two(matrix)
    return matrix
  
  # 在矩阵的随机位置且值为0的地方赋值为2
  def add_two(matrix):
    a = random.randint(0, len(matrix)-1)
    b = random.randint(0, len(matrix)-1)
    while matrix[a][b] != 0:
      a = random.randint(0, len(matrix)-1)
      b = random.randint(0, len(matrix)-1)
    matrix[a][b] = 2
    return matrix
  
  # 矩阵翻转
  def reverse(matrix):
    new = []
    for i in range(len(matrix)):
      new.append([])
      for j in range(len(matrix[0])):
        new[i].append(matrix[i][len(matrix[0])-j-1])
    return new

  # 矩阵转置
  def transpose(matrix):
    new = []
    for i in range(len(matrix[0])):
      new.append([])
      for j in range(len(matrix)):
        new[i].append(matrix[j][i])
    return new

  # 矩阵元素往左移动
  def cover_up(mat):
    new = []
    for j in range(GRID_LEN):
      partial_new = []
      for i in range(GRID_LEN):
        partial_new.append(0)
      new.append(partial_new)
    done = False

    for i in range(GRID_LEN):
      count = 0
      for j in range(GRID_LEN):
        if mat[i][j] != 0:
          new[i][count] = mat[i][j]
          if j != count:
            done = True
          count += 1
    return new, done

  # 矩阵元素合并
  def merge(mat, done):
    score = 0
    for i in range(GRID_LEN):
      for j in range(GRID_LEN-1):
        if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
          mat[i][j] *= 2
          mat[i][j+1] = 0
          score += mat[i][j]
          done = True
    return mat, done, score
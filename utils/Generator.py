import random
import constants as c

class Generator:
  # 生成一个随机整数，范围在0到c.GRID_LEN - 1之间
  def gen():
    return random.randint(0, c.GRID_LEN - 1)
  
  # 随机位置生成下一个数字
  def generate_next(matrix):
    index = (Generator.gen(), Generator.gen())
    while matrix[index[0]][index[1]] != 0:
      index = (Generator.gen(), Generator.gen())
    matrix[index[0]][index[1]] = 2
  
  # 生成一个n行n列的矩阵，每个元素初始化为0
  def generate_matrix(n):
    matrix = []
    for i in range(n):
      matrix.append([0] * n)
    return matrix
class GameState:
  # 检查矩阵中是否有2048
  def __check_win(matrix):
    for i in range(len(matrix)):
      for j in range(len(matrix[0])):
        if matrix[i][j] == 2048:
          return True
    return False

  # 检查矩阵中是否有0
  def __check_empty(matrix):
    for i in range(len(matrix)):
      for j in range(len(matrix[0])):
        if matrix[i][j] == 0:
          return True
    return False

  # 检查相邻的单元格是否相同(只能用于4x4的矩阵)
  def __check_same_adjacent(matrix):
    # 检查每一行的第一个和最后一个单元格
    for i in range(len(matrix)):
      if matrix[i][0] == matrix[i][1]:
        return True
      if matrix[i][len(matrix[0])-1] == matrix[i][len(matrix[0])-2]:
        return True
    # 检查每一列的第一个和最后一个单元格
    for j in range(len(matrix[0])):
      if matrix[0][j] == matrix[1][j]:
        return True
      if matrix[len(matrix)-1][j] == matrix[len(matrix)-2][j]:
        return True
    return False

  # 检查相邻的单元格是否相同
  def __check_same_adjacentN(matrix, n):
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
  def game_state(matrix):
    if GameState.__check_win(matrix):
      return 'win'
    if GameState.__check_empty(matrix):
      return 'not over'
    if GameState.__check_same_adjacent(matrix):
      return 'not over'
    return 'lose'
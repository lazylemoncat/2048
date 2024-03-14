from tkinter import Frame, Label, CENTER
from utils.Matrix import Matrix
from utils.GameState import GameState
from constants.color import *
from constants.size import *
from constants.font import *
from utils.KeyListener import KeyListener

# 继承Frame类
class Game(Frame):
  def __init__(self):
    Frame.__init__(self)
    # 运行标识
    self.isRunning = True
    # 游戏状态
    self.state = "not over"
    # 网格布局
    self.grid()
    # 设置窗口标题
    self.master.title('2048')
    # 绑定键盘事件
    self.master.bind("<Key>", self.key_down)
    # 初始化格子文本组件
    self.grid_cells = []
    # 初始化网格
    self.init_grid()
    # 初始化矩阵
    self.matrix = Matrix.new_game(GRID_LEN)
    # 刷新格子文本组件
    self.update_grid_cells()
    # 运行
    self.mainloop()

  def init_grid(self):
    background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
    # 网格布局
    background.grid()
    for row in range(GRID_LEN):
      grid_row = []
      for col in range(GRID_LEN):
        cell = Frame(
          background,
          width=SIZE / GRID_LEN,
          height=SIZE / GRID_LEN
        )
        cell.grid(
          row=row,
          column=col,
          padx=GRID_PADDING,
          pady=GRID_PADDING
        )
        text = Label(
          master=cell,
          text="",
          justify=CENTER,
          font=FONT,
          width=5,
          height=2
        )
        text.grid()
        grid_row.append(text)
      self.grid_cells.append(grid_row)

  # 更新格子文本组件
  def update_grid_cells(self):
    for row in range(GRID_LEN):
      for col in range(GRID_LEN):
        num = self.matrix[row][col]
        if num == 0:
          self.grid_cells[row][col].configure(text="",bg=BACKGROUND_COLOR_CELL_EMPTY)
        else:
          self.grid_cells[row][col].configure(
            text=str(num),
            bg=BACKGROUND_COLOR_DICT[num],
            fg=CELL_COLOR_DICT[num]
          )
    # 刷新视觉显示
    self.update_idletasks()

  # 键盘事件
  def key_down(self, event):
    self.matrix, done = KeyListener.key_down(event, self.matrix)
    if done:
      self.matrix = Matrix.add_two(self.matrix)
      self.update_grid_cells()
      self.state = GameState.game_state(self.matrix)
      if self.state == 'win':
        self.isRunning = False
        self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Win!", bg=BACKGROUND_COLOR_CELL_EMPTY)
      if self.state == 'lose':
        self.isRunning = False
        self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)

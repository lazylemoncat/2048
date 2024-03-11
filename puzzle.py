from tkinter import Frame, Label, CENTER
import logic
import constant as c
from constants.color import *
from utils.KeyListener import KeyListener

# 继承Frame类
class GameGrid(Frame):
  def __init__(self):
    Frame.__init__(self)
    # 网格布局
    self.grid()
    # 设置窗口标题
    self.master.title('2048')
    # 绑定键盘事件
    self.master.bind("<Key>", self.key_down)

    self.grid_cells = []
    self.init_grid()
    self.matrix = logic.new_game(c.GRID_LEN)
    self.update_grid_cells()
    # 运行
    self.mainloop()

  def init_grid(self):
    background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
    # 网格布局
    background.grid()

    for row in range(c.GRID_LEN):
      grid_row = []
      for col in range(c.GRID_LEN):
        cell = Frame(
          background,
          bg=BACKGROUND_COLOR_CELL_EMPTY,
          width=c.SIZE / c.GRID_LEN,
          height=c.SIZE / c.GRID_LEN
        )
        cell.grid(
          row=row,
          column=col,
          padx=c.GRID_PADDING,
          pady=c.GRID_PADDING
        )
        text = Label(
          master=cell,
          text="",
          bg=BACKGROUND_COLOR_CELL_EMPTY,
          justify=CENTER,
          font=c.FONT,
          width=5,
          height=2)
        text.grid()
        grid_row.append(text)
      self.grid_cells.append(grid_row)

  def update_grid_cells(self):
    for i in range(c.GRID_LEN):
      for j in range(c.GRID_LEN):
        new_number = self.matrix[i][j]
        if new_number == 0:
          self.grid_cells[i][j].configure(text="",bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        else:
          self.grid_cells[i][j].configure(
            text=str(new_number),
            bg=BACKGROUND_COLOR_DICT[new_number],
            fg=CELL_COLOR_DICT[new_number]
          )
    self.update_idletasks()

  def key_down(self, event):
    self.matrix, done = KeyListener.key_down(event, self.matrix)
    if done:
      self.matrix = logic.add_two(self.matrix)
      self.update_grid_cells()
      if logic.game_state(self.matrix) == 'win':
        self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Win!", bg=BACKGROUND_COLOR_CELL_EMPTY)
      if logic.game_state(self.matrix) == 'lose':
        self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)

game_grid = GameGrid()
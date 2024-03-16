from tkinter import Frame, Label, CENTER
from constants.color import *
from constants.size import *
from constants.font import *

class Window(Frame):
  def __init__(self, key_down):
    Frame.__init__(self)
    # 网格布局
    self.grid()
    # 设置窗口标题
    self.master.title('2048')
    # 绑定键盘事件
    self.master.bind("<Key>", key_down)
    # 初始化格子文本组件
    self.__grid_cells = []
    # 初始化网格
    self.__init_grid()

  def getGridCells(self):
    return self.__grid_cells
  
  def __init_grid(self):
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
      self.__grid_cells.append(grid_row)
    
  # 更新格子文本组件
  def update_grid_cells(self, matrix):
    for row in range(GRID_LEN):
      for col in range(GRID_LEN):
        num = matrix[row][col]
        if num == 0:
          self.__grid_cells[row][col].configure(text="",bg=BACKGROUND_COLOR_CELL_EMPTY)
        else:
          self.__grid_cells[row][col].configure(
            text=str(num),
            bg=BACKGROUND_COLOR_DICT[num],
            fg=CELL_COLOR_DICT[num]
          )
    # 刷新视觉显示
    self.update_idletasks()
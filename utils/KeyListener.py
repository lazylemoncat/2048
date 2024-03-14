from utils.Move import Move
from constants.key import *

class KeyListener:
  def key_down(event, matrix):
    key = event.keysym
    commands = {
      KEY_UP: Move.up,
      KEY_DOWN: Move.down,
      KEY_LEFT: Move.left,
      KEY_RIGHT: Move.right,
      KEY_UP_ALT1: Move.up,
      KEY_DOWN_ALT1: Move.down,
      KEY_LEFT_ALT1: Move.left,
      KEY_RIGHT_ALT1: Move.right,
      KEY_UP_ALT2: Move.up,
      KEY_DOWN_ALT2: Move.down,
      KEY_LEFT_ALT2: Move.left,
      KEY_RIGHT_ALT2: Move.right,
    }
    print(event, matrix)
    done = False
    if key == KEY_QUIT: 
      exit()
    elif key in commands:
      matrix, done = commands[key](matrix)
    return matrix, done
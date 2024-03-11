import logic
from constants.key import *

class KeyListener:
  def key_down(event, matrix):
    key = event.keysym
    commands = {
      KEY_UP: logic.up,
      KEY_DOWN: logic.down,
      KEY_LEFT: logic.left,
      KEY_RIGHT: logic.right,
      KEY_UP_ALT1: logic.up,
      KEY_DOWN_ALT1: logic.down,
      KEY_LEFT_ALT1: logic.left,
      KEY_RIGHT_ALT1: logic.right,
      KEY_UP_ALT2: logic.up,
      KEY_DOWN_ALT2: logic.down,
      KEY_LEFT_ALT2: logic.left,
      KEY_RIGHT_ALT2: logic.right,
    }
    print(event, matrix)
    done = False
    if key == KEY_QUIT: 
      exit()
    elif key in commands:
      matrix, done = commands[key](matrix)
    return matrix, done
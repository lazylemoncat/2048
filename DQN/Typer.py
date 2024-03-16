from time import sleep
import time
from pynput.keyboard import Key, Controller
import threading
import random

class Typer:
  def __init__(self):
    self.__keyboard = Controller()
  
  def type(self, char):
    self.__keyboard.type(char)
    sleep(0.5)


def main():
  typer = Typer()
  sleep(1)
  while True:
    actions = ['w', 'a', 's', 'd']
    char = actions[random.randint(0, 3)]
    typer.type(char)
    print(char)
    time.sleep(0.01)

if __name__ == "__main__":
  thread_a = threading.Thread(target=main)

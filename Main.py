from Game import Game
import threading

if __name__ == "__main__":
  # 创建线程
  # thread_a = threading.Thread(target=main)
  # thread_a.start()
  game = Game()
  game.run()
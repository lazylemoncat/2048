import time
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingLogger:
  def __init__(self):
    self.iteration = 0
    self.iterations = []
    self.scores = []

  def clear(self):
    self.iteration = 0
    self.iterations.clear()
    self.scores.clear()

  def log(self, score, iteration=None):
    if iteration is None:
      self.iteration = self.iteration + 1
    else:
      self.iteration = iteration
    self.iterations.append(self.iteration)
    self.scores.append(score)

  def plot(self):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))

    plt.plot(self.iterations, self.scores, marker='o', linestyle='-', color='b', label='Scores')
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Training Progress', fontsize=16)
    
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    # 时间
    plt.savefig(f"imgs/training-{time.strftime('%Y-%m-%d%H_%M_%S', time.localtime())}.png")
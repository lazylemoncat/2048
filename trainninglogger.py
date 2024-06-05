import matplotlib.pyplot as plt
from datetime import datetime

class TrainingLogger:
    def __init__(self):
        self.times = []
        self.iterations = []
        self.scores = []

    def log(self, time, iteration, score):
        self.times.append(time)
        self.iterations.append(iteration)
        self.scores.append(score)

    def plot(self):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Iteration', color='tab:blue')
        ax1.plot(self.times, self.iterations, 'b-', label='Iterations')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Score', color='tab:red')  # we already handled the x-label with ax1
        ax2.plot(self.times, self.scores, 'r-', label='Scores')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Training Progress')
        plt.show()

# 示例用法
logger = TrainingLogger()

# 假设我们在某个时间点记录了迭代次数和分数
logger.log(datetime.now(), 10, 50)
logger.log(datetime.now(), 20, 60)
logger.log(datetime.now(), 30, 70)

# 最后生成图表
logger.plot()

import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils import *



def make_bar_plot(values: np.array):
    # Compute pie slices
    N = len(values)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = np.pi * 2 / N
    colors = [plt.cm.viridis(0.6)]
    ax = plt.subplot(projection='polar')
    ax.bar(theta, values, width=width, bottom=0.0, color=colors, alpha=0.5)
    plt.show()


def add_danger(current_danger, new_danger, angle):
    N = len(current_danger)
    offset = (N - int(angle * N / (2 * np.pi) - N / 2)) % N
    print(offset)
    danger = current_danger - \
        np.concatenate((new_danger[offset:], new_danger[:offset]))
    return danger

def create_danger():
    mu = 0
    sigma = N/40
    x = np.linspace(-N/2, N/2, N)
    y2 = stats.norm.pdf(x, mu, sigma)*60
    # plt.plot(x, y2)
    # plt.show()

class PreyBaseline:
    def __init__(self, environment):
        self.N = 21
        self.current_decision = LEFT
        self.step = 0

    def decide(self, observation):
        self.step += 1
        
        if self.step % 20:
            return

        if self.current_decision == LEFT:
            self.current_decision = DOWN
        elif self.current_decision == DOWN:
            self.current_decision = RIGHT
        elif self.current_decision == RIGHT:
            self.current_decision = UP
        elif self.current_decision == UP:
            self.current_decision = LEFT
    
    def act(self):
        return self.current_decision
        
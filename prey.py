import scipy.stats as stats
import matplotlib.pyplot as plt

from agent import Agent
from utils import *


def add_danger(current_danger, new_danger, angle):
    N = len(current_danger)
    offset = (N - int(angle * N / (2 * np.pi) - N / 2)) % N
    # print(offset)
    danger = current_danger - \
             np.concatenate((new_danger[offset:], new_danger[:offset]))
    return danger


class PreyBaseline(Agent):
    def __init__(self):
        super().__init__()
        self.step = 0

    def reset(self):
        self.current_decision = LEFT
        self.step = 0

    def observe(self, observation):
        pass

    def decide(self):
        self.step += 1

        if self.step % 20:
            return self.current_decision

        if self.current_decision == LEFT:
            self.current_decision = DOWN
        elif self.current_decision == DOWN:
            self.current_decision = RIGHT
        elif self.current_decision == RIGHT:
            self.current_decision = UP
        elif self.current_decision == UP:
            self.current_decision = LEFT

        return self.current_decision


class PreyDangerCircle(Agent):
    def __init__(self):
        super().__init__()
        self.N = 41
        self.danger = np.ones(self.N) * 20

    def decide(self):
        index_to_action = {
            0: RIGHT,
            1: UP,
            2: LEFT,
            3: DOWN
        }

        up_total = sum(self.danger[round(self.N / 8):round(3 * self.N / 8)])
        left_total = sum(self.danger[round(3 * self.N / 8):round(5 * self.N / 8)])
        down_total = sum(self.danger[round(5 * self.N / 8):round(7 * self.N / 8)])
        right_total = sum(list(self.danger[0:round(self.N / 8)]) +
                          list(self.danger[round(7 * self.N / 8):self.N]))

        action_index = np.argmax([right_total, up_total, left_total, down_total])
        self.current_decision = index_to_action[action_index]

        return self.current_decision

    def observe(self, observation):
        self.calculate_danger(observation["self_pos"], observation["other_agents_pos"], observation["obstacle_pos"])

    def calculate_danger(self, coords, predator_positions, obstacle_positions):
        self.danger = np.ones(self.N) * 20
        for adv_coords in predator_positions:
            adv_danger = self.get_distance_normal(get_distance(adv_coords), 1)
            angle = get_agent_angle(adv_coords)
            self.danger = add_danger(self.danger, adv_danger, angle)

        for obs_coords in obstacle_positions:
            adv_danger = self.get_distance_normal(get_distance(obs_coords), 1)
            angle = get_agent_angle(obs_coords)
            self.danger = add_danger(self.danger, adv_danger, angle)

        center_distance = get_distance(coords)
        angle = get_agent_angle(coords)
        if center_distance < 1:
            center_danger = self.get_distance_normal(1 - get_distance(coords), 1)
        else:
            center_danger = self.get_distance_normal(0.1, 1)

        self.danger = add_danger(self.danger, center_danger, angle)
        # self.showDanger()

    def showDanger(self):
        # Compute pie slices
        theta = np.linspace(0.0, 2 * np.pi, self.N, endpoint=False)
        width = np.pi * 2 / self.N
        colors = [plt.cm.viridis(0.6)]
        ax = plt.subplot(projection='polar')
        ax.bar(theta, self.danger, width=width, bottom=0.0, color=colors, alpha=0.5)
        ax.set_ylim(0, 20)
        plt.show()

    def get_distance_normal(self, distance, magnitude):
        x = np.linspace(-self.N / 2, self.N / 2, self.N)
        return stats.norm.pdf(x, 0, 1 / distance) * 1 / distance * self.N * magnitude

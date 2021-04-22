import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from agent import Agent
from utils import *


def add_danger(current_danger, new_danger, angle):
    N = len(current_danger)
    offset = (N - int(angle * N / (2 * np.pi) - N / 2)) % N
    # print(offset)
    danger = current_danger - \
             np.concatenate((new_danger[offset:], new_danger[:offset]))
    return danger


class Prey(Agent, ABC):
    def __init__(self):
        super().__init__()

    def observe(self, observation):
        self.obstacle_distances.clear()
        self.predator_distances.clear()
        self.prey_positions.clear()
        self.prey_velocities.clear()

        self.vel = (observation[0], observation[1])
        self.coords = (observation[2], observation[3])

        i = 4

        for j in range(NUM_OBSTACLES):
            self.obstacle_distances.append((observation[i], observation[i + 1]))
            i += 2

        for j in range(NUM_ADVERSARIES):
            self.predator_distances.append((observation[i], observation[i + 1]))
            i += 2

        for j in range(NUM_GOOD - 1):
            self.prey_positions.append((observation[i], observation[i + 1]))
            i += 2

        for j in range(NUM_GOOD - 1):
            self.prey_velocities.append((observation[i], observation[i + 1]))
            i += 2

        if i != len(observation):
            raise Exception("Too many inputs")


class PreyBaseline(Prey):
    def __init__(self):
        super().__init__()
        self.N = 21
        self.step = 0

    def decide(self):
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


class PreyDangerCircle(Prey):
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
        theta = np.linspace(0.0, 2 * np.pi, self.N, endpoint=False)

        up_total = sum(self.danger[round(self.N / 8):round(3 * self.N / 8)])
        left_total = sum(self.danger[round(3 * self.N / 8):round(5 * self.N / 8)])
        down_total = sum(self.danger[round(5 * self.N / 8):round(7 * self.N / 8)])
        right_total = sum(list(self.danger[0:round(self.N / 8)]) +
                          list(self.danger[round(7 * self.N / 8):self.N]))

        action_index = np.argmax([right_total, up_total, left_total, down_total])
        self.current_decision = index_to_action[action_index]

    def observe(self, observation):
        super().observe(observation)
        self.calculate_danger()

    def calculate_danger(self):
        self.danger = np.ones(self.N) * 20
        for adv_coords in self.predator_distances:
            adv_danger = self.get_distance_normal(get_distance(adv_coords), 1)
            angle = get_agent_angle(adv_coords)
            self.danger = add_danger(self.danger, adv_danger, angle)

        for obs_coords in self.obstacle_distances:
            adv_danger = self.get_distance_normal(get_distance(obs_coords), 1)
            angle = get_agent_angle(obs_coords)
            self.danger = add_danger(self.danger, adv_danger, angle)

        center_distance = get_distance(self.coords)
        angle = get_agent_angle(self.coords)
        if center_distance < 1:
            center_danger = self.get_distance_normal(1 - get_distance(self.coords), 1)
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
        # ax.set_theta_zero_location("N")
        plt.show()

    def get_distance_normal(self, distance, magnitude):
        x = np.linspace(-self.N / 2, self.N / 2, self.N)
        return stats.norm.pdf(x, 0, 1 / distance) * 1 / distance * self.N * magnitude

    def act(self):
        return self.current_decision

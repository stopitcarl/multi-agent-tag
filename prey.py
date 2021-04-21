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
        # TODO better way to choose max
        index_to_action = {
            0: RIGHT,
            1: UP,
            2: LEFT,
            3: DOWN
        }
        theta = np.linspace(0.0, 2 * np.pi, self.N, endpoint=False)
        angle = theta[np.argmax(self.danger)]

        # self.showDanger()
        # print("best angle:", np.rad2deg(angle % (2 * np.pi)))

        action_index = round(angle/(np.pi/2)) % 4
        self.current_decision = index_to_action[action_index]

    def observe(self, observation):
        super().observe(observation)
        self.calculate_danger()

    def calculate_danger(self):
        self.danger = np.ones(self.N) * 20
        for adv_coords in self.predator_distances:

            distance = np.linalg.norm(np.array(adv_coords) - np.array(self.coords))
            adv_danger = self.get_distance_normal(distance)

            angle = np.arctan2(adv_coords[1]-self.coords[1], adv_coords[0]-self.coords[0])
            # print("adversary at:", np.rad2deg(angle % (2 * np.pi)), "degrees")
            self.danger = add_danger(self.danger, adv_danger, angle)

        distance_to_center = np.linalg.norm(np.array((0, 0)) - np.array(self.coords))
        center_danger = self.get_distance_normal(distance_to_center)
        angle = (np.arctan2(0 - self.coords[1], 0 - self.coords[0]) + np.pi) % (2*np.pi)
        self.danger = add_danger(self.danger, center_danger, angle)

    def showDanger(self):
        # Compute pie slices
        theta = np.linspace(0.0, 2 * np.pi, self.N, endpoint=False)
        width = np.pi * 2 / self.N
        colors = [plt.cm.viridis(0.6)]
        ax = plt.subplot(projection='polar')
        ax.bar(theta, self.danger, width=width, bottom=0.0, color=colors, alpha=0.5)
        plt.show()

    def get_distance_normal(self, distance):
        x = np.linspace(-self.N / 2, self.N / 2, self.N)
        return stats.norm.pdf(x, 0, distance) * 60

    def act(self):
        return self.current_decision
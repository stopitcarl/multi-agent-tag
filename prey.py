import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import utils
from agent import Agent
from utils import *


class PreyBaseline(Agent):
    def __init__(self):
        super().__init__()
        self.N = 41
        self.danger = np.ones(self.N) * 20
        self.theta = np.linspace(0.0, 2 * np.pi, self.N, endpoint=False)
        self.width = np.pi * 2 / self.N
        self.colors = [plt.cm.viridis(0.6)]
        self.speed = [0, 0]
        self.rects = None

    def observe(self, observation):
        observation = utils.decode(observation)
        self.calculate_danger(
            observation["self_pos"], observation["other_agents_pos"], observation["obstacle_pos"])
        self.speed = observation["self_vel"]
        # print(observation["self_pos"])
        # print(self.speed)

    def decide(self):
        angle = self.decide_angle()
        self.current_decision = self.decide_speed(angle)
        return self.current_decision

    def calculate_danger(self, coords, predator_positions, obstacle_positions):
        self.danger = np.ones(self.N) * 20
        # Calculate danger for adversaries
        for adv_coords in predator_positions:
            # Calculate the angle of the adversary
            angle = get_angle(adv_coords)
            # Calculate the distance of the adversary
            dist = get_distance(adv_coords) + 0.2
            adv_danger = self.get_distance_normal(
                0.2 / dist, 0.5 / dist)
            # Add adversary danger to consideration
            self.danger = self.add_danger(adv_danger, angle)
        # Calculate danger for obstacles
        for obs_coords in obstacle_positions:
            adv_danger = self.get_distance_normal(
                get_distance(obs_coords), 0.5)
            angle = get_angle(obs_coords)
            self.danger = self.add_danger(adv_danger, angle)

        center_distance = get_distance(coords) + 0.2
        angle = get_angle(coords)
        center_danger = self.get_distance_normal(
            0.3, (center_distance ** (2)))

        self.danger = self.add_danger(center_danger, angle)
        # self.show_danger()

    def decide_angle(self):
        directions = 8
        sums = np.zeros(directions)
        step = self.N // directions
        start = -step // 2 + 1
        for i in range(directions):
            sums[i] = self.danger_sum(self.danger, start, step)
            start += step
        angles = [np.pi / (directions / 2) * i for i in range(directions)]
        angle = angles[np.argmax(sums)]
        return angle

    def decide_speed(self, angle):
        current = Vector.from_point(*self.speed)
        desired = Vector(angle)
        decision = desired * 2.5 - current
        return decision.to_action()

    def danger_sum(self, danger, start, step):
        if start < 0:
            return np.sum(danger[start:]) + np.sum(danger[:start + step])
        else:
            return sum(danger[start:start + step])

    def add_danger(self, new_danger, angle):
        N = self.N
        offset = (N - int(angle * N / (2 * np.pi) - N / 2)) % N
        # print(offset)
        danger = self.danger - \
                 np.concatenate((new_danger[offset:], new_danger[:offset]))
        return danger

    def show_danger(self):
        ax = plt.subplot()
        ax = plt.subplot(projection='polar')
        self.rects = ax.bar(self.theta, self.danger, width=self.width,
                            bottom=0.0, color=self.colors, alpha=0.5)
        # ax.set_ylim(0, 20)

        # def animate():
        #     for rect, h in zip(self.rects, self.danger):
        #         rect.set_height(h)
        #     return self.rects

        # ani = animation.FuncAnimation(fig, animate, blit=True, interval=100,
        #                               repeat=True)
        plt.show()

    def get_distance_normal(self, distance, magnitude):
        x = np.linspace(-self.N / 2, self.N / 2, self.N)
        return stats.norm.pdf(x, 0, 2 / distance) * 1 / distance * self.N * magnitude


class Vector:
    def __init__(self, angle):
        self.x = np.cos(angle)
        self.y = np.sin(angle)

        norm = np.sqrt(self.x ** 2 + self.y ** 2)
        self.x /= norm
        self.y /= norm

    def to_action(self):
        if abs(self.x) >= abs(self.y):
            return LEFT if self.x < 0 else RIGHT
        else:
            return DOWN if self.y < 0 else UP

    @classmethod
    def from_point(cls, x, y):
        angle = atan2(y, x) % (2 * np.pi)
        return cls(angle)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        if x == y == 0:
            return self
        return Vector.from_point(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        if x == y == 0:
            return self
        return Vector.from_point(x, y)

    def __mul__(self, other):
        self.x *= other
        self.y *= other
        return self

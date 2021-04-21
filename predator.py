import numpy as np
from abc import ABC, abstractmethod
from agent import Agent
from utils import *


class Predator(Agent, ABC):
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

        for j in range(NUM_ADVERSARIES-1):
            self.predator_distances.append((observation[i], observation[i + 1]))
            i += 2

        for j in range(NUM_GOOD):
            self.prey_positions.append((observation[i], observation[i + 1]))
            i += 2

        for j in range(NUM_GOOD):
            self.prey_velocities.append((observation[i], observation[i + 1]))
            i += 2

        if i != len(observation):
            raise Exception("Too many inputs")


class PredatorBaseline(Predator):
    def __init__(self):
        super().__init__()

    def decide(self):
        angle = get_agent_angle(self.prey_positions[0])
        if angle >= 0:
            if angle < np.pi/4:
                self.current_decision = UP
            elif angle < 3*np.pi/4:
                self.current_decision = RIGHT
            else:
                self.current_decision = DOWN
        else:
            if angle > -np.pi/4:
                self.current_decision = UP
            elif angle > -3*np.pi/4:
                self.current_decision = LEFT
            else:
                self.current_decision = DOWN

    def act(self):
        return self.current_decision

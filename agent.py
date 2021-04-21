from abc import ABC, abstractmethod
from enum import Enum
from utils import *


class Agent(ABC):
    def __init__(self):
        self.vel = (0, 0)
        self.coords = (0, 0)
        self.obstacle_distances = []
        self.predator_distances = []
        self.prey_positions = []
        self.prey_velocities = []
        self.current_decision = LEFT

    @abstractmethod
    def observe(self, observation):
        pass

    @abstractmethod
    def decide(self):
        pass

    @abstractmethod
    def act(self):
        pass
from abc import ABC, abstractmethod
from utils import *


class Agent(ABC):
    def __init__(self):
        self.current_decision = LEFT

    @abstractmethod
    def observe(self, observation):
        pass

    @abstractmethod
    def decide(self):
        pass
import utils
from agent import Agent
from utils import *


class PredatorBaseline(Agent):
    def __init__(self, n_predators, n_prey, n_obstacles):
        super().__init__()
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.n_obstacles = n_obstacles
        self.angle = 0

    def observe(self, observation):
        observation = utils.decode(observation, self.n_predators, self.n_obstacles)
        self.angle = get_angle(observation["other_agents_pos"][-1])

    def decide(self):
        if np.pi / 4 <= self.angle < 3 * np.pi / 4:
            self.current_decision = UP
        elif 3 * np.pi / 4 <= self.angle < 5 * np.pi / 4:
            self.current_decision = LEFT
        elif 5 * np.pi / 4 <= self.angle < 7 * np.pi / 4:
            self.current_decision = DOWN
        elif 7 * np.pi / 4 <= self.angle < 2 * np.pi or 0 <= self.angle < np.pi / 4:
            self.current_decision = RIGHT

        return self.current_decision

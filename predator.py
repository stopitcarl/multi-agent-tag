import numpy as np

from utils import *


class PredatorBaseline:
    def __init__(self, environment):
        self.current_decision = UP

    def decide(self, observation):
        angle = get_agent_angle(get_other_pos(observation)[-1])
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

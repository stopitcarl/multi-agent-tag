from math import atan2

NUM_GOOD = 1
NUM_ADVERSARIES = 2
NUM_OBSTACLES = 2
MAX_CYCLES = 300

# observations
# [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
SELF_VEL_X = 0
SELF_VEL_Y = 1
SELF_POS_X = 2
SELF_POS_Y = 3
ENEMY_POS_X = 4
ENEMY_POS_Y = 5
ENEMY_VEL_X = 6
ENEMY_VEL_Y = 7


# action space
NO_ACTION = 0
LEFT = 1
RIGHT = 2
DOWN = 3
UP = 4


def get_own_vel(observation: list):
    return observation[:2]


def get_own_pos(observation: list):
    return observation[2:4]


def get_obstacles_pos(observation: list):
    return [observation[4 + i * 2:4 + i * 2 + 2] for i in range(NUM_OBSTACLES)]


def get_other_pos(observation: list):
    return [observation[4 + 2 * NUM_OBSTACLES + 2 * i:4 + 2 * NUM_OBSTACLES + 2 * i + 2] for i in range(NUM_ADVERSARIES)]


def get_prey_vel(observation: list):
    return observation[-2:0]


def get_agent_angle(position):
    return atan2(position[0], position[1])

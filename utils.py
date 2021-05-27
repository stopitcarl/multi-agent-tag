from math import atan2
import numpy as np

# action space
NO_ACTION = 0
LEFT = 1
RIGHT = 2
DOWN = 3
UP = 4


def decode(observation: list, n_predators, n_obstacles):
    return {
        'self_vel': observation[:2],
        'self_pos': observation[2:4],
        'obstacle_pos': [observation[4 + i * 2:4 + i * 2 + 2] for i in range(n_obstacles)],
        'other_agents_pos': [observation[4 + 2 * n_obstacles + 2 * i:4 + 2 * n_obstacles + 2 * i + 2] for i in
                             range(n_predators)],
        # will only work if observation is from predator, otherwise it will be garbage
        'prey_velocity': observation[-2:]
    }


def get_own_vel(observation: list):
    return observation[:2]


def get_own_pos(observation: list):
    return observation[2:4]


def get_obstacles_pos(observation: list, n_obstacles):
    return [observation[4 + i * 2:4 + i * 2 + 2] for i in range(n_obstacles)]


def get_other_pos(observation: list, n_predators, n_obstacles):
    return [observation[4 + 2 * n_obstacles + 2 * i:4 + 2 * n_obstacles + 2 * i + 2] for i in
            range(n_predators)]


def get_prey_vel(observation: list):
    return observation[-2:0]


def get_angle(position):
    return atan2(position[1], position[0]) % (2 * np.pi)


def get_distance(coords):
    return np.linalg.norm(np.array((0, 0)) - np.array(coords))

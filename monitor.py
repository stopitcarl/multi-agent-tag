import numpy as np
import utils


class Monitor:
    def __init__(self, *, n_prey, n_predators, n_obstacles: int, steps_per_game: int, prey_names: list,
                 predator_names: list):
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.n_obstacles = n_obstacles
        self.steps_per_game = steps_per_game
        self.prey_names = prey_names
        self.predator_names = predator_names

        self.steps = 0
        self.game = 0
        self.game_step = 0

        self.prey_positions = {name: [] for name in prey_names}
        self.predator_positions = {name: [] for name in predator_names}

        self.predator_prey_positions = {name: [] for name in predator_names}
        self.collisions = {self.game: []}

    def log(self, observations, rewards):
        for name in self.prey_names:
            self.prey_positions[name] += [utils.get_own_pos(observations[name])]

        for name in self.predator_names:
            self.predator_positions[name] += [utils.get_own_pos(observations[name])]
            self.predator_prey_positions[name] += [utils.get_other_pos(observations[name])[-2:]]

        if rewards[self.predator_names[0]] > 0:
            self.collisions[self.game] += [self.game_step]

        self.steps += 1
        self.game_step += 1
        if self.game_step >= self.steps_per_game:
            self.game_step = 0
            self.game += 1
            self.collisions[self.game] = []

    def average_distance_predator_prey(self):
        return np.average([
            utils.get_distance(self.predator_prey_positions[name][step])
            for step in range(self.steps)
            for name in self.predator_names])

    def average_steps_until_capture(self):
        return np.average([self.collisions[game][0] for game in range(self.game) if len(self.collisions[game]) > 0])

    def average_number_collisions(self):
        return np.average([len(self.collisions[game]) for game in range(self.game)])

    def average_distance_to_center(self):
        return np.average([
            utils.get_distance(self.prey_positions[name][step])
            for step in range(self.steps)
            for name in self.prey_names])

    def stats(self):
        print("-------------Statistics-------------")
        print("Average distance to center: ", self.average_distance_to_center())
        print("Average distance predators-prey: ", self.average_distance_predator_prey())
        print("Average time until capture: ", self.average_steps_until_capture())
        print("Average number of collisions: ", self.average_number_collisions())

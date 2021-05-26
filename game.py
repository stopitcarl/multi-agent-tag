import time

import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from prey import PreyBaseline
from dqn import PreyQLearning, PredatorQLearning
from utils import *
from monitor import Monitor

# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=SEED)

# setup agents
agents = {f'adversary_{i}': PredatorBaseline() for i in range(NUM_ADVERSARIES)}
agents.update({f'agent_{i}': PreyBaseline() for i in range(NUM_GOOD)})
monitor = Monitor(n_prey=NUM_GOOD, n_predators=NUM_ADVERSARIES, n_obstacles=NUM_OBSTACLES, steps_per_game=MAX_CYCLES,
                  prey_names=[f"agent_{i}" for i in range(NUM_GOOD)],
                  predator_names=[f"adversary_{i}" for i in range(NUM_ADVERSARIES)])

# game loop
for game in range(NUM_GAMES):
    state = env.reset()
    for _ in range(MAX_CYCLES):
        # env.render()

        actions = {}

        for name, agent in agents.items():
            agent.observe(state[name])
            actions[name] = agent.decide()

        state, rewards, done, _ = env.step(actions)
        monitor.log(state, rewards)

env.close()

monitor.stats()

import time

import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from prey import PreyBaseline
from utils import *

# setup enviornment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=42)
env.reset()

# setup actions
actions = {agent: 0 for agent in env.agents}
actions

# setup agents
agents = {}
agents['adversary_0'] = PredatorBaseline(env)
agents['adversary_1'] = PredatorBaseline(env)
agents['agent_0'] = PreyBaseline(env)

# game loop
for step in range(MAX_CYCLES):
    env.render()
    observations, rewards, dones, infos = env.step(actions)

    time.sleep(0.01)

    # decision process
    for name, agent in agents.items():
        agent.decide(observations[name])

    # action process
    for name, agent in agents.items():
        actions[name] = agent.act()

print("done")
env.close()

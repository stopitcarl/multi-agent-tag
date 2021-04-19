import time

import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from prey import PreyBaseline
from utils import *

env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=42)
env.reset()

# setup actions
actions = {agent: 0 for agent in env.agents}
actions

# setup agents
adversary_0 = PredatorBaseline(env)
adversary_1 = PredatorBaseline(env)
agent_0 = PreyBaseline(env)


for step in range(MAX_CYCLES):
    env.render()
    observations, rewards, dones, infos = env.step(actions)

    time.sleep(0.05)

    # decision process
    adversary_0.decide(observations['adversary_0'])
    adversary_1.decide(observations['adversary_1'])
    agent_0.decide(observations['agent_0'])

    # action process
    actions['adversary_0'] = adversary_0.act()
    actions['adversary_1'] = adversary_1.act()
    actions['agent_0'] = agent_0.act()

print("done")
env.close()

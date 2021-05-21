import time

import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from prey import PreyBaseline, PreyDangerCircle
from utils import *

# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=SEED)
print(env.reset())

# setup actions
actions = {agent: 0 for agent in env.agents}

# setup agents
agents = {f'adversary_{i}': PredatorBaseline() for i in range(NUM_ADVERSARIES)}
agents['agent_0'] = PreyDangerCircle()

# game loop
for step in range(MAX_CYCLES):
    env.render()
    observations, rewards, dones, infos = env.step(actions)

    time.sleep(0.01)

    # decision process
    for name, agent in agents.items():
        obs = decode(observations[name])
        agent.observe(obs)

    # action process
    for name, agent in agents.items():
        actions[name] = agent.decide()

    #actions['agent_0'] = NO_ACTION    

print("done")
env.close()

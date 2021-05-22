import time

import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline, PredatorSimpleGeneticAlgorithm
from prey import PreyBaseline, PreyDangerCircle
from utils import *

# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.reset()

# setup actions
actions = {agent: 0 for agent in env.agents}

# setup agents
agents = {
    'adversary_0': PredatorSimpleGeneticAlgorithm(),
    'adversary_1': PredatorSimpleGeneticAlgorithm(),
    'agent_0': PreyDangerCircle()
}

agents['adversary_0'].train(env=env,
                            prey=agents['agent_0'],
                            num_gens=3,
                            pop_size=4,
                            num_parents=2,
                            mutation_rate=0.2,
                            to_save=1,
                            verbose=1)
agents['adversary_0'].load_model()
agents['adversary_1'].load_model()

env.seed(seed=SEED)
env.reset()

input("Press enter to play with the best model")

# game loop
for step in range(MAX_CYCLES):
    env.render()
    observations, rewards, dones, infos = env.step(actions)
    time.sleep(0.1)

    # decision process
    for name, agent in agents.items():
        obs = decode(observations[name])
        agent.observe(obs)

    # action process
    for name, agent in agents.items():
        actions[name] = agent.decide()

print("done")
env.close()

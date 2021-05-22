import time

import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from PredatorMutationBasedGeneticController import PredatorMutationBasedGeneticController
from prey import PreyBaseline, PreyDangerCircle
from utils import *

# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.reset()

controller = PredatorMutationBasedGeneticController(
    env,
    PreyBaseline(),
    10,
    10,
    200,
    ranked_selection=False,
    scheduled_mutation=False
)
controller.train()

# # setup actions
# actions = {agent: 0 for agent in env.agents}
#
# # setup agents
# agents = {
#     'adversary_0': PredatorSimpleGeneticAlgorithm(),
#     'adversary_1': PredatorSimpleGeneticAlgorithm(),
#     'agent_0': PreyDangerCircle()
# }
#
# env.seed(seed=SEED)
# env.reset()
#
# input("Press enter to play with the best model")
#
# # game loop
# for step in range(MAX_CYCLES):
#     env.render()
#     observations, rewards, dones, infos = env.step(actions)
#     time.sleep(0.1)
#
#     # decision process
#     for name, agent in agents.items():
#         obs = decode(observations[name])
#         agent.observe(obs)
#
#     # action process
#     for name, agent in agents.items():
#         actions[name] = agent.decide()
#
# print("done")
# env.close()

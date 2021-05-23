from pettingzoo.mpe import simple_tag_v2

from PredatorMutationBasedGeneticController import PredatorMutationBasedGeneticController
from prey import PreyBaseline, PreyDangerCircle
from utils import *

env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.reset()

controller = PredatorMutationBasedGeneticController(
    env,
    PreyBaseline(),
    1000,
    100,
    100,
    ranked_selection=False,
    scheduled_mutation=True
)
controller.train()
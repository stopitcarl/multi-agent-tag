import time

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
    1000,
    100,
    200,
    ranked_selection=False,
    scheduled_mutation=True
)
controller.load_model("models/MGNN.h5")

# setup actions
actions = {agent: 0 for agent in env.agents}

# setup agents
prey = PreyBaseline()

env.seed(seed=SEED)
env.reset()

# game loop
for step in range(MAX_CYCLES):
    env.render()
    observations, rewards, dones, infos = env.step(actions)
    time.sleep(0.1)

    prey.observe(decode(observations["agent_0"]))
    controller.observe(decode(observations["adversary_0"]))

    predatorActions = controller.decide()

    actions = {
        "agent_0": prey.decide(),
        "adversary_0": predatorActions[0],
        "adversary_1": predatorActions[1],
        "adversary_2": predatorActions[2]
    }

    # for name, agent in agents.items():
    #     obs = decode(observations[name])
    #     agent.observe(obs)

    # for name, agent in agents.items():
    #     actions[name] = agent.decide()

print("done")
env.close()

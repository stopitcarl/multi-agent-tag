import time

from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from PredatorMutationBasedGeneticController import PredatorMutationBasedGeneticController
from prey import PreyBaseline, PreyDangerCircle
from agent import SmartAgent, SmartController
from utils import *

# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.reset()

# setup actions
actions = {agent: 0 for agent in env.agents}

controller = SmartController("models/PyGad13.h5")

# setup agents
prey = SmartAgent("models/DangerZone.h5")
# prey = PreyDangerCircle()

# env.seed(seed=SEED)
env.reset()

# game loop
for step in range(MAX_CYCLES):
    env.render()
    observations, rewards, dones, infos = env.step(actions)
    time.sleep(0.1)

    prey.observe(observations["agent_0"])
    controller.observe(observations["adversary_0"])

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

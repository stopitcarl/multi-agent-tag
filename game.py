import time
import matplotlib.pyplot as plt
import numpy.random as rand
from tqdm import tqdm
from pettingzoo.mpe import simple_tag_v2

from predator import PredatorBaseline
from prey import PreyBaseline, PreyDangerCircle
from utils import *

# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=SEED)
observations = env.reset()

# setup actions
actions = {agent: 0 for agent in env.agents}

# setup agents
agents = {f'adversary_{i}': PredatorBaseline() for i in range(NUM_ADVERSARIES)}
agents['agent_0'] = PreyDangerCircle()

plot_rewards = []

for _ in tqdm(range(100), unit='games'):
    game_reward = 0
    observations = env.reset()
    # game loop
    for step in range(MAX_CYCLES):
        #env.render()
        #time.sleep(0.02)

        # decision process
        for name, agent in agents.items():
            obs = decode(observations[name])
            agent.observe(obs)

        # action process
        for name, agent in agents.items():
            actions[name] = agent.decide()

        observations, rewards, dones, infos = env.step(actions)

        #actions['agent_0'] = NO_ACTION
        game_reward += rewards['agent_0']
    plot_rewards.append(game_reward)

plt.plot(np.arange(100), plot_rewards)
plt.show()

print("done")
env.close()

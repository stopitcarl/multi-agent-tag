
import time
import numpy as np
import numpy.random as rand
from pettingzoo.mpe import simple_tag_v2
from predator import PredatorBaseline
from prey import PreyBaseline, PreyDangerCircle
from utils import *

import tensorflow as tf
from statistics import median, mean
from collections import Counter


# setup enviornment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=SEED)

# data gathering config
goal_steps = MAX_CYCLES
score_requirement = 50
initial_games = 10
agent_of_interest = 'agent_0'

# setup agents
agents = {f'adversary_{i}': PredatorBaseline() for i in range(NUM_ADVERSARIES)}
agents['agent_0'] = PreyDangerCircle()

# setup actions
actions = {agent: 0 for agent in agents.keys()}

# [OBS, MOVES]
training_data = []
# all scores:
scores = []
# just the scores that met our threshold:
accepted_scores = []

start_time = time.perf_counter()
# iterate through however many games we want:
for _ in range(initial_games):
    score = 0
    # moves specifically from this environment:
    game_memory = []
    # previous observation that we saw
    prev_observation = []

    # reset env to play again
    observations = env.reset()

    # game loop
    for _ in range(goal_steps):
        # decision process
        for name, agent in agents.items():
            obs = decode(observations[name])
            agent.observe(obs)
        # action process
        for name, agent in agents.items():
            actions[name] = agent.decide()
        # do it!
        observations, rewards, dones, infos = env.step(actions)

        # notice that the observation is returned FROM the action
        # so we'll store the previous observation here, pairing
        # the prev observation to the action we'll take.
        if len(prev_observation):
            game_memory.append([prev_observation, actions[agent_of_interest]])
        prev_observation = observations[agent_of_interest]
        score += rewards[agent_of_interest]

    accepted_scores.append(score)
    for data in game_memory:
        # convert to one-hot (this is the output layer for our neural network)
        output = tf.keras.utils.to_categorical(data[1], num_classes=5)
        # saving our training data
        training_data.append([data[0], output])
    # save overall scores
    scores.append(score)

env.close()

end_time = time.perf_counter()

print("Took", end_time - start_time, "seconds")
print((end_time - start_time) / initial_games, "seconds per game")

# just in case you wanted to reference later
training_data_save = np.array(training_data, dtype=object)
np.save('saved.npy', training_data_save)

# some stats here, to further illustrate the neural network magic!
print('Average accepted score:', mean(accepted_scores))
print('Median score for accepted scores:', median(accepted_scores))
# print(Counter(accepted_scores))

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os

from agent import Agent
from utils import *


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
# Minimum number of steps in a memory to start training
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = MAX_CYCLES  # Terminal states (end of episodes)
MODEL_NAME = 'firt_try_dqn'
MIN_REWARD = -140  # For model save

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# Training settings
ROUNDS = 100

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        #self._write_logs(stats, self.step)
        for name, value in stats.items():
            tf.summary.scalar(name, value, step=self.step)


# Agent class
class DQNAgent(Agent):
    def __init__(self, env):

        # agent's environment
        # self.env = env

        # agent's environment
        self.name = "agent_0"

        # Main model
        self.model = self.create_model(env)

        # Target network
        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
        # Used to save agent's state and make decisions
        self.current_state = None

    def create_model(self, env):
        model = tf.keras.Sequential(name='multi_layer')

        # OBSERVATION_SPACE_SHAPE = (14,) for the prey
        model.add(tf.keras.layers.Input(
            env.observation_spaces['agent_0'].shape))

        model.add(tf.keras.layers.Dense(128, activation='relu', name='hidden'))

        # model.add(Conv2D(256, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        # ACTION_SPACE_SIZE = how many choices (5)
        model.add(tf.keras.layers.Dense(
            env.action_spaces['agent_0'].n, activation='linear', name='output'))

        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def observe(self, current_state):
        self.current_state = current_state

    def decide(self):
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(self.get_qs(self.current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_spaces['agent_0'].n)
        return action

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0]
                                   for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


# setup environment
env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.seed(seed=SEED)
env.reset()

# setup actions
actions = {agent: 0 for agent in env.agents}

# setup agents
agents = {f'adversary_{i}': PredatorBaseline() for i in range(NUM_ADVERSARIES)}
agents['agent_0'] = DQNAgent(env)


agent = agents['agent_0']

# Iterate over episodes
for episode in tqdm(range(1, ROUNDS + 1), ascii=True, unit='rounds'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state['agent_0']))
        else:
            # Get random action
            action = np.random.randint(0, env.action_spaces['agent_0'].n)

        new_state, rewards, dones, infos = env.step(actions)
        #new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += rewards['agent_0']

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory(
            (current_state['agent_0'], action, rewards['agent_0'], new_state['agent_0']))
        agent.train(done)

        current_state = new_state
        step += 1

        if step == MAX_CYCLES or rewards['adversary_0'] > 10:
            done = True

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(
            ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(
            reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

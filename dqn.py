import time

from agent import Agent
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pettingzoo.mpe import simple_tag_v2


class PreyQLearning(Agent):
    def __init__(self, *, n_predators: int, n_prey: int, n_obstacles: int, model_path: str = None):
        super().__init__()
        input_size = 4 + n_obstacles * 2 + (n_prey + n_predators - 1) * 2
        self.net = DeepQNetwork(input_size=input_size, model_path=model_path)
        self.state = []

    def observe(self, observation):
        self.state = observation

    def decide(self):
        return self.net.act(self.state)

    def train_decide(self):
        return self.net.get_action(self.state)

    def transition(self, state, action, new_state, reward, done):
        return self.net.update_transition_table(state, action, new_state, reward, done)

    def save(self, path):
        self.net.save(path)


class PredatorQLearning(Agent):

    def __init__(self, *, n_predators: int, n_prey: int, n_obstacles: int, model_path: str = None):
        super().__init__()
        input_size = 4 + n_obstacles * 2 + (n_prey + n_predators - 1) * 2 + n_prey * 2
        self.net = DeepQNetwork(input_size=input_size, model_path=model_path)
        self.state = []

    def observe(self, observation):
        self.state = observation

    def decide(self):
        return self.net.act(self.state)

    def train_decide(self):
        return self.net.get_action(self.state)

    def transition(self, state, action, new_state, reward, done):
        return self.net.update_transition_table(state, action, new_state, reward, done)

    def save(self, path):
        self.net.save(path)


class DeepQNetwork:
    def __init__(self, *, input_size, model_path):
        # Model
        if model_path:
            self.model = self.load_model(path=model_path)
        else:
            self.model = self.create_model(input_size)
            self.target = self.create_model(input_size)
            self._update_target()

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.loss = keras.losses.Huber()

        # Transition table
        self.actions = []
        self.states = []
        self.new_states = []
        self.rewards = []
        self.dones = []

        # Control
        self.gamma = 0.95
        self.steps = 0
        self.random_steps = 0
        self.epsilon_random_steps = 200000
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.train_after_n_actions = 10
        self.training_size = 1024
        self.update_target_after_n_actions = 1000

        self.max_length = 100000
        self.delete_size = 100

    @staticmethod
    def create_model(input_size):
        inputs = layers.Input(shape=(input_size,))
        layer1 = layers.Dense(64, activation="relu")(inputs)
        layer2 = layers.Dense(64, activation="relu")(layer1)
        actions = layers.Dense(5, activation="linear")(layer2)
        return keras.Model(inputs=inputs, outputs=actions)

    @staticmethod
    def load_model(path):
        return keras.models.load_model(path)

    def act(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probabilities = self.model(state_tensor, training=False)
        return np.argmax(action_probabilities[0])

    def get_action(self, state):

        if self.steps < self.random_steps or self.epsilon > np.random.rand(1)[0]:
            action = np.random.randint(5)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probabilities = self.model(state_tensor, training=False)
            action = np.argmax(action_probabilities[0])

        self.epsilon -= (1 - self.min_epsilon) / self.epsilon_random_steps
        self.epsilon = max(self.epsilon, self.min_epsilon)

        return action

    def update_transition_table(self, state, action, new_state, reward, done):
        self.steps += 1
        self.states.append(state)
        self.actions.append(action)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.dones.append(done)

        if self.steps % 10000 == 0:
            print(f"epsilon: {self.epsilon}")

        # Update model
        if self.steps % self.train_after_n_actions == 0 and len(self.dones) > self.training_size:
            self._update_weights()

        # Update target model
        if self.steps % self.update_target_after_n_actions == 0:
            self._update_target()

    def _update_weights(self):

        # Get training sample
        sample = np.random.choice(range(len(self.dones)), size=self.training_size)

        states_sample = np.array([self.states[i] for i in sample])
        actions_sample = np.array([self.actions[i] for i in sample])
        new_states_sample = np.array([self.new_states[i] for i in sample])
        rewards_sample = np.array([self.rewards[i] for i in sample])
        dones_sample = tf.convert_to_tensor([float(self.dones[i]) for i in sample])

        future_rewards = self.target.predict(new_states_sample)
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
        updated_q_values = updated_q_values * (1 - dones_sample) - dones_sample

        masks = tf.one_hot(actions_sample, 5)
        with tf.GradientTape() as tape:
            q_values = self.model(states_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def _update_target(self):
        print("Updating target model")
        self.target.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save(f"{path}_model")
        self.target.save(f"{path}_target")

    def _free_up_memory(self):
        del self.steps[:self.delete_size]
        del self.states[:self.delete_size]
        del self.actions[:self.delete_size]
        del self.new_states[:self.delete_size]
        del self.rewards[:self.delete_size]


def train(n_prey: int, n_predators: int, n_obstacles: int, cycles_per_game: int, n_games: int, render: int,
          save_path: str):
    env = simple_tag_v2.parallel_env(num_good=n_prey, num_adversaries=n_predators, num_obstacles=n_obstacles,
                                     max_cycles=cycles_per_game)
    env.seed(42)

    preys = {f"agent_{i}": PreyQLearning(n_predators=n_predators, n_prey=n_prey,
                                         n_obstacles=n_obstacles) for i in range(n_prey)}
    predators = {
        f"adversary_{i}": PredatorQLearning(n_predators=n_predators, n_prey=n_prey,
                                            n_obstacles=n_obstacles)
        for i in range(n_predators)}

    steps = 0

    for game in range(n_games):
        print(f"Starting game {game}")
        state = env.reset()
        game_reward = {f"agent_{i}": 0.0 for i in range(n_prey)}
        game_reward.update({f"adversary_{i}": 0.0 for i in range(n_predators)})
        for _ in range(cycles_per_game):
            if steps > render:
                env.render()

            actions = {}

            for name, agent in preys.items():
                agent.observe(state[name])
                actions[name] = agent.train_decide()

            for name, agent in predators.items():
                agent.observe(state[name])
                actions[name] = agent.train_decide()

            new_state, rewards, done, _ = env.step(actions)

            for name in preys.keys():
                game_reward[name] += rewards[name]
            for name in predators.keys():
                game_reward[name] += rewards[name]

            for name, agent in preys.items():
                agent.transition(state[name], actions[name], new_state[name], rewards[name], done[name])
            for name, agent in predators.items():
                agent.transition(state[name], actions[name], new_state[name], rewards[name], done[name])

            state = new_state
            steps += 1
            if steps % 50000 == 0:
                for name, predator in preys.items():
                    predator.save(f"{save_path}/{steps}_{name}")
                for name, predator in predators.items():
                    predator.save(f"{save_path}/{steps}_{name}")
        print(f"Accumulated {game_reward}")

    if steps > render:
        env.close()


def test(n_prey: int, n_predators: int, n_obstacles: int, cycles_per_game: int, n_games: int, prey_model_path: list,
         predator_model_path: list):
    env = simple_tag_v2.parallel_env(num_good=n_prey, num_adversaries=n_predators, num_obstacles=n_obstacles,
                                     max_cycles=cycles_per_game)
    env.seed(42)

    preys = {f"agent_{i}": PreyQLearning(n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
                                         model_path=prey_model_path[i]) for i in range(n_prey)}
    predators = {f"adversary_{i}": PredatorQLearning(n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles,
                                                     model_path=predator_model_path[i]) for i in range(n_predators)}

    for game in range(n_games):
        state = env.reset()
        for _ in range(cycles_per_game):
            env.render()
            time.sleep(0.2)

            actions = {}

            for name, agent in preys.items():
                agent.observe(state[name])
                actions[name] = agent.decide()

            for name, agent in predators.items():
                agent.observe(state[name])
                actions[name] = agent.decide()

            state, rewards, done, _ = env.step(actions)
    env.close()


"""
# This takes a long time to run, make sure to not override already trained nets
train(n_prey=1, n_predators=3, n_obstacles=2, cycles_per_game=100, n_games=5000, render=np.inf, save_path="example")
"""

"""
test(n_prey=1, n_predators=3, n_obstacles=2, cycles_per_game=100, n_games=100,
     prey_model_path=["dqn_models/agent_0_model"],
     predator_model_path=["dqn_models/adversary_0_model", "dqn_models/adversary_1_model",
                          "dqn_models/adversary_2_model"])
"""

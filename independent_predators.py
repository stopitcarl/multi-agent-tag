from agent import Agent
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pettingzoo.mpe import simple_tag_v2
from utils import decode
from prey import PreyDangerCircle


class PredatorIndependentController(Agent):

    def __init__(self, *, n_predators: int, n_prey: int, n_obstacles: int):
        super().__init__()
        self.net = PredatorIndependentController.DeepQNetwork(n_predators=n_predators,
                                                              n_prey=n_prey,
                                                              n_obstacles=n_obstacles)
        self.state = []

    def observe(self, observation):
        self.state = observation

    def decide(self):
        return self.net.get_action(self.state)

    def transition(self, state, action, new_state, reward):
        return self.net.update_transition_table(state, action, new_state, reward)

    def save(self, path):
        self.net.save(path)

    class DeepQNetwork:
        def __init__(self, *, n_predators: int, n_prey: int, n_obstacles: int):
            input_size = 4 + n_obstacles * 2 + (n_prey + n_predators - 1) * 2 + n_prey * 2

            # Model
            self.model = self.create_model(input_size)
            self.target = self.create_model(input_size)
            self.loss_function = keras.losses.Huber()
            self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

            # Transition table
            self.actions = []
            self.states = []
            self.new_states = []
            self.rewards = []

            # Control
            self.gamma = 0.99
            self.steps = 0
            self.random_steps = 50000
            self.epsilon_random_steps = 100000
            self.epsilon = 1
            self.min_epsilon = 0.1
            self.train_after_n_actions = 5
            self.training_size = 50
            self.update_target_after_n_actions = 10000

            self.max_length = 500000
            self.delete_size = 100

        @staticmethod
        def create_model(input_size):
            inputs = layers.Input(shape=input_size)
            layer1 = layers.Dense(128, activation="relu")(inputs)
            layer2 = layers.Dense(64, activation="relu")(layer1)
            actions = layers.Dense(5, activation="linear")(layer2)
            return keras.Model(inputs=inputs, outputs=actions)

        def get_action(self, state):

            if self.steps < self.random_steps or self.epsilon > np.random.rand(1)[0]:
                action = np.random.randint(5)
            else:
                tensor = tf.convert_to_tensor(state)
                tensor = tf.expand_dims(tensor, 0)
                action_probabilities = self.model(tensor, training=False)
                action = tf.argmax(action_probabilities[0]).numpy()

            self.epsilon -= (1 - self.min_epsilon) / self.epsilon_random_steps
            self.epsilon = max(self.epsilon, self.min_epsilon)

            return action

        def update_transition_table(self, state, action, new_state, reward):
            self.steps += 1
            self.states.append(state)
            self.actions.append(action)
            self.new_states.append(new_state)
            self.rewards.append(reward)

            # Update model
            if self.steps % self.train_after_n_actions == 0 and len(self.states) > self.training_size:
                self._update_weights()

            # Update target model
            if self.steps % self.update_target_after_n_actions == 0:
                self._update_target()

        def _update_weights(self):

            # Get training sample
            sample = np.random.choice(range(len(self.states)), size=self.training_size)

            states_sample = np.array([self.states[i] for i in sample])
            actions_sample = np.array([self.actions[i] for i in sample])
            new_states_sample = np.array([self.new_states[i] for i in sample])
            rewards_sample = np.array([self.rewards[i] for i in sample])

            # Update q-values
            future_rewards = self.target.predict(new_states_sample)
            updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

            # Train
            mask = tf.one_hot(actions_sample, 5)
            with tf.GradientTape() as tape:
                q_values = self.model(states_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, mask))
                loss = self.loss_function(updated_q_values, q_action)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        def _update_target(self):
            print("Updating target")
            self.target.set_weights(self.model.get_weights())

        def save(self, path):
            self.model.save(path)

        def _free_up_memory(self):
            del self.steps[:self.delete_size]
            del self.states[:self.delete_size]
            del self.actions[:self.delete_size]
            del self.new_states[:self.delete_size]
            del self.rewards[:self.delete_size]


def train(*, n_prey: int, prey_controllers, n_predators: int, n_obstacles: int, cycles_per_game: int, n_games: int,
          render: int):
    env = simple_tag_v2.parallel_env(num_good=n_prey, num_adversaries=n_predators, num_obstacles=n_obstacles,
                                     max_cycles=cycles_per_game)

    preys = {f"agent_{i}": prey_controllers[i] for i in range(n_prey)}
    predators = {
        f"adversary_{i}": PredatorIndependentController(n_predators=n_predators, n_prey=n_prey, n_obstacles=n_obstacles)
        for i in range(n_predators)}

    steps = 0
    for game in range(n_games):
        print(f"Starting game {game}")
        env.seed(np.random.randint(n_games))
        state = env.reset()
        for _ in range(cycles_per_game):
            if steps > render:
                env.render()

            actions = {}

            for name, agent in preys.items():
                obs = decode(state[name])
                agent.observe(obs)
                actions[name] = agent.decide()

            for name, agent in predators.items():
                agent.observe(state[name])
                actions[name] = agent.decide()

            new_state, rewards, _, _ = env.step(actions)

            # Give positive rewards to all predators
            positive_reward = 0
            for name, agent in rewards.items():
                if name.startswith("adversary") and rewards[name] > 0:
                    positive_reward += rewards[name]
            if positive_reward > 0:
                print(f"hit target {positive_reward}")

            for name, agent in predators.items():
                agent.transition(state[name], actions[name], new_state[name], rewards[name] + positive_reward)

            state = new_state
            steps += 1
    if steps > render:
        env.close()

    return predators.values()


models = train(n_prey=1, prey_controllers=[PreyDangerCircle()], n_predators=3, n_obstacles=2, cycles_per_game=1000,
               n_games=1000, render=100000)

i = 0
for model in models:
    model.save(f"models/best_{i}")
    i += 1

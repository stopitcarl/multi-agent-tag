import itertools

from agent import Agent
from utils import *
import tensorflow as tf
from tensorflow import keras
import multiprocessing as mp


class PredatorBaseline(Agent):
    def __init__(self):
        super().__init__()
        self.angle = 0

    def observe(self, observation):
        self.angle = get_agent_angle(observation["other_agents_pos"][-1])

    def decide(self):
        if np.pi/4 <= self.angle < 3*np.pi/4:
            self.current_decision = UP
        elif 3*np.pi/4 <= self.angle < 5*np.pi/4:
            self.current_decision = LEFT
        elif 5*np.pi/4 <= self.angle < 7*np.pi/4:
            self.current_decision = DOWN
        elif 7*np.pi/4 <= self.angle < 2*np.pi or 0 <= self.angle < np.pi/4:
            self.current_decision = RIGHT

        return self.current_decision


class PredatorSimpleGeneticAlgorithm(Agent):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()

    def load_model(self):
        self.model = tf.keras.models.load_model("models/PredatorSimpleGeneticAlgorithmModel.h5", compile=False)

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(10 + NUM_ADVERSARIES * 2, input_shape=(10 + NUM_ADVERSARIES * 2,), activation="relu",
                               kernel_initializer='random_normal', bias_initializer='zeros'),
            keras.layers.Dense(20, activation="relu",
                               kernel_initializer='random_normal', bias_initializer='zeros'),
            keras.layers.Dense(5, activation="softmax",
                               kernel_initializer='random_normal', bias_initializer='zeros')
        ])

        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    def train(self, env, prey, num_gens, pop_size, num_parents, mutation_rate, to_save, verbose=0):

        def fitness(model):
            score = 0
            env.reset()

            actions = {agent: 0 for agent in env.agents}

            for step in range(MAX_CYCLES):
                observations, rewards, dones, infos = env.step(actions)

                for name in actions.keys():
                    if "agent" not in name:
                        score += rewards[name]

                for name in actions.keys():
                    obs = decode(observations[name])
                    if "agent" in name:
                        prey.observe(obs)
                        actions[name] = prey.decide()
                    else:
                        actions[name] = predict(model, obs)

            return score

        def predict(model, observation):
            neural_input = np.concatenate([
                observation["self_vel"],
                observation["self_pos"],
                np.concatenate(observation["obstacle_pos"]),
                np.concatenate(observation["other_agents_pos"]),
                observation["prey_velocity"]
            ])

            neural_input = np.atleast_2d(neural_input)

            return np.argmax(model.predict(neural_input))

        def make_new_gen(parents):
            def crossover(mother, father):
                child_model = tf.keras.models.clone_model(mother)

                for i, layer in enumerate(mother.layers):
                    new_weights_for_layer = []
                    father_weights = father.layers[i].get_weights()
                    mother_weights = mother.layers[i].get_weights()

                    for j, weight_array in enumerate(layer.get_weights()):
                        save_shape = weight_array.shape
                        mother_weight_array = mother_weights[j].reshape(-1)
                        father_weight_array = father_weights[j].reshape(-1)
                        one_dim_weight = weight_array.reshape(-1)

                        for k, weight in enumerate(one_dim_weight):
                            one_dim_weight[k] = np.random.choice([mother_weight_array[k], father_weight_array[k]])

                        new_weight_array = one_dim_weight.reshape(save_shape)
                        new_weights_for_layer.append(new_weight_array)

                    child_model.layers[i].set_weights(new_weights_for_layer)

                return child_model

            def mutate(model, rate):
                for i, layer in enumerate(model.layers):
                    new_weights_for_layer = []

                    for weight_array in layer.get_weights():
                        save_shape = weight_array.shape
                        one_dim_weight = weight_array.reshape(-1)

                        for j, weight in enumerate(one_dim_weight):
                            if np.random.uniform(0, 1) <= rate:
                                one_dim_weight[j] = np.random.uniform(0, 2) - 1

                        new_weight_array = one_dim_weight.reshape(save_shape)
                        new_weights_for_layer.append(new_weight_array)

                    model.layers[i].set_weights(new_weights_for_layer)

                return model

            def breed(couple):
                child_model = crossover(couple[0]["model"], couple[1]["model"])
                child_model = mutate(child_model, mutation_rate)
                return {"model": child_model, "fitness": -np.inf}

            couples = []
            for parent in parents:
                couples.append((parent, np.random.choice(parents)))

            # pool = mp.Pool(mp.cpu_count())
            # children = [pool.apply(breed, args=couples) for _ in range(pop_size)]
            children = parents[-to_save:]
            while len(children) < pop_size:
                couple = couples[np.random.randint(0, len(couples))]
                np.append(children, breed(couple))

            return np.array(children)

        current_pool = []

        # Create population
        for _ in range(pop_size):
            current_pool.append({"model": self.create_model(), "fitness": -np.inf})

        current_pool = np.array(current_pool)

        for i in range(num_gens):
            if verbose > 0:
                print("\nGeneration " + str(i) + ":\n")

            # Evaluate fitness
            for j, agent in enumerate(current_pool):
                if verbose > 0:
                    print("Agent " + str(j) + ":")
                agent["fitness"] = fitness(agent["model"])

                if verbose > 0:
                    print("Fitness:", agent["fitness"])

            # Choose top x
            top_agents = current_pool[np.argsort([x["fitness"] for x in current_pool])[-num_parents:]]

            if i % 20 == 0:
                best_model = top_agents[-1]["model"]
                best_model.save("models/PredatorSimpleGeneticAlgorithmModel_gen" + str(i) + ".h5")

            # Cross and Mutate
            if i < num_gens-1:
                current_pool = make_new_gen(top_agents)

        best_model = current_pool[np.argsort([x["fitness"] for x in current_pool])[-1]][0]["model"]
        best_model.save("models/PredatorSimpleGeneticAlgorithmModel.h5")
        self.model = best_model

    def observe(self, observation):
        neural_input = np.concatenate([
            observation["self_vel"],
            observation["self_pos"],
            np.concatenate(observation["obstacle_pos"]),
            np.concatenate(observation["other_agents_pos"]),
            observation["prey_velocity"]
        ])

        neural_input = np.atleast_2d(neural_input)

        self.current_decision = np.argmax(self.model.predict(neural_input))

    def decide(self):
        return self.current_decision

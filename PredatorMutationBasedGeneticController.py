import numpy as np
from utils import *
from tensorflow import keras


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(12 + (NUM_ADVERSARIES - 1) * 2, input_shape=(12 + (NUM_ADVERSARIES - 1) * 2,),
                           activation="relu",
                           kernel_initializer='random_normal', bias_initializer='zeros'),
        keras.layers.Dense(20, activation="relu",
                           kernel_initializer='random_normal', bias_initializer='zeros'),
        keras.layers.Dense(5 * NUM_ADVERSARIES, activation="softmax",
                           kernel_initializer='random_normal', bias_initializer='zeros')
    ])

    return model


def gaussian_perturbation(sigma):
    return np.random.normal(0, abs(sigma))


def encode_neural_input(observation):
    neural_input = np.concatenate([
        observation["self_vel"],
        observation["self_pos"],
        np.concatenate(observation["obstacle_pos"]),
        np.concatenate(observation["other_agents_pos"]),
        observation["prey_velocity"]
    ])
    return np.atleast_2d(neural_input)


class NeuralNetwork:

    def __init__(self, perturbation_func, mutation_prob):
        self.perturbation_func = perturbation_func
        self.mutation_prob = mutation_prob
        self.mutationSSP = 0
        self.mutation = 0
        self.model = create_model()
        self.fitness = 0

    def compute_fitness(self, env, prey):
        score = 0
        observations = env.reset()

        actions = {agent: 0 for agent in env.agents}

        for step in range(MAX_CYCLES):
            pred_obs = self.predict(decode(observations["adversary_0"]))

            for name in actions.keys():
                obs = decode(observations[name])

                if "agent" in name:
                    prey.observe(obs)
                    actions[name] = prey.decide()
                else:
                    actions[name] = pred_obs.pop(0)

            observations, rewards, dones, infos = env.step(actions)

            for name in actions.keys():
                if "agent" not in name:
                    score += rewards[name]

        self.fitness = score

    def predict(self, observation):
        prediction = self.model.predict(encode_neural_input(observation))[0]
        return [np.argmax(prediction[i:i + 5]) for i in range(0, len(prediction), 5)]

    def mutate(self, SSP):
        self.mutationSSP = self.perturbation_func(SSP)
        self.mutation += self.perturbation_func(self.mutationSSP)

        for i, layer in enumerate(self.model.layers):
            new_weights_for_layer = []

            for weight_array in layer.get_weights():
                save_shape = weight_array.shape
                one_dim_weight = weight_array.reshape(-1)

                for j, weight in enumerate(one_dim_weight):
                    if np.random.uniform() <= self.mutation_prob:
                        one_dim_weight[j] += self.perturbation_func(self.mutation)

                new_weight_array = one_dim_weight.reshape(save_shape)
                new_weights_for_layer.append(new_weight_array)

            self.model.layers[i].set_weights(new_weights_for_layer)


class PredatorMutationBasedGeneticController:
    def __init__(self, env, prey, num_gens, pop_size, SSP, ranked_selection=False, scheduled_mutation=False):
        self.env = env
        self.prey = prey
        self.num_gens = num_gens
        self.pop_size = pop_size
        self.SSP = SSP
        self.ranked_selection = ranked_selection
        self.scheduled_mutation = scheduled_mutation
        self.population = []
        self.model = None
        self.decisions = [LEFT, LEFT, LEFT]

    def initialize_population(self):
        self.population = [
            NeuralNetwork(
                perturbation_func=gaussian_perturbation,
                mutation_prob=0.01
            )
            for _ in range(self.pop_size)
        ]

    def mutate_population(self, first=False):
        if self.scheduled_mutation and not first:
            start = 0.01
            end = 0.05
            step = (end - start) / self.pop_size
            self.population.sort(key=lambda x: x.fitness)

            for nn in self.population:
                nn.mutation_prob = start
                start += step

        for nn in self.population:
            nn.mutate(self.SSP)

    def roulette_wheel_selection(self):
        pop_fitness = sum([individual.fitness for individual in self.population])

        if pop_fitness == 0:
            individual_probs = [1 / self.pop_size for _ in range(self.pop_size)]
        else:
            individual_probs = [individual.fitness / pop_fitness for individual in self.population]

        new_pop = []
        while len(new_pop) < self.pop_size:
            new_pop.append(np.random.choice(self.population, p=individual_probs))

        self.population = new_pop

    def ranked_based_selection(self):
        self.population.sort(key=lambda x: x.fitness)
        individual_probs = np.linspace(0, 1, self.pop_size)

        new_pop = []
        while len(new_pop) < self.pop_size:
            new_pop.append(np.random.choice(self.population, p=individual_probs))

        self.population = new_pop

    def select_next_parents(self):
        if self.ranked_selection:
            self.ranked_based_selection()
        else:
            self.roulette_wheel_selection()

    def train(self):
        self.initialize_population()
        self.mutate_population(True)

        for gen in range(self.num_gens):
            for nn in self.population:
                nn.compute_fitness(self.env, self.prey)

            self.print_stats(gen)

            if gen % 10 == 0:
                self.save_best(str(gen))

            self.select_next_parents()

            self.mutate_population()

        self.save_best()
        self.load_model()

    def print_stats(self, gen):
        fitnesses = [child.fitness for child in self.population]
        print(
            "Gen " + str(gen) +
            ", best: " + str(max(fitnesses)) +
            ", average: " + str(sum(fitnesses) / self.pop_size)
        )

    def save_best(self, gen=""):
        self.population.sort(key=lambda x: x.fitness)
        self.population[-1].model.save("models/MGNN" + str(gen) + ".h5")

    def load_model(self, path="models/MGNN.h5"):
        self.model = keras.models.load_model(path, compile=False)

    def observe(self, observation):
        prediction = self.model.predict(encode_neural_input(observation))[0]
        self.decisions = [np.argmax(prediction[i:i + 5]) for i in range(0, len(prediction), 5)]

    def decide(self):
        return self.decisions

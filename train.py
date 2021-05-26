from pettingzoo.mpe import simple_tag_v2
from tensorflow import keras
from utils import *
import numpy as np
import pygad.kerasga
from prey import PreyBaseline
from agent import Agent


class SmartAgent(Agent):
    def __init__(self, model_path):
        # Main model
        super().__init__()
        self.model = keras.models.load_model(model_path)

    def observe(self, current_state):
        self.current_state = current_state

    def decide(self):
        action = np.argmax(
            self.model.predict(
                np.array(self.current_state).reshape(-1, *self.current_state.shape))[0])
        return action

    def is_smart(self):
        return True


def create_model():
    inputs = keras.Input(shape=(12 + (NUM_ADVERSARIES - 1) * 2,))
    dense_input = keras.layers.Dense(12 + (NUM_ADVERSARIES - 1) * 2, activation="tanh",
                                     kernel_initializer='random_normal', bias_initializer='random_normal')
    dense_hidden1 = keras.layers.Dense(15, activation="tanh",
                                       kernel_initializer='random_normal', bias_initializer='random_normal')
    dense_hidden2 = keras.layers.Dense(10, activation="tanh",
                                       kernel_initializer='random_normal', bias_initializer='random_normal')

    dense_outputs = []
    for _ in range(NUM_ADVERSARIES):
        dense_outputs.append(keras.layers.Dense(5, activation="softmax",
                                                kernel_initializer='random_normal', bias_initializer='random_normal'))

    x = dense_input(inputs)
    x = dense_hidden1(x)
    x = dense_hidden2(x)

    outputs = []
    for i in range(NUM_ADVERSARIES):
        outputs.append(dense_outputs[i](x))

    model = keras.Model(inputs=inputs, outputs=outputs, name='MGNN_brain')
    return model


def fitness_func(solution, solution_idx):
    global env, model, prey, gen_count

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    score = 0
    # prey.reset()
    env.seed(gen_count)
    gen_count += 1
    observations = env.reset()

    actions = {agent: 0 for agent in env.agents}

    for step in range(MAX_CYCLES):
        predictions = model.predict(np.atleast_2d(observations["adversary_0"]))
        predator_actions = [np.argmax(prediction[0]) for prediction in predictions]

        for name in actions.keys():
            # env.render()
            obs = decode(observations[name])

            if "agent" in name:
                prey.observe(observations[name])
                actions[name] = prey.decide()
            else:
                score -= get_distance(obs["other_agents_pos"][-1])
                actions[name] = predator_actions.pop(0)

        observations, rewards, dones, infos = env.step(actions)

        score += rewards["adversary_0"]

    return score


def callback_generation(ga_instance):
    solution, solution_fitness, _ = ga_instance.best_solution()
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=solution_fitness))
    print("Average    = {average}".format(
        average=sum(ga_instance.last_generation_fitness) / len(ga_instance.last_generation_fitness)))

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=create_model(), weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    model.save("models/PyGad" + str(ga_instance.generations_completed) + ".h5")


gen_count = 0

env = simple_tag_v2.parallel_env(
    num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES)
env.reset()

prey = SmartAgent("models/DangerZone.h5")

model = keras.models.load_model("models/PyGadGlitch.h5", compile=False)
# model = create_model()
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=50)

num_generations = 250
num_parents_mating = 25
initial_population = keras_ga.population_weights
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 40
keep_parents = -1

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(weights=model_weights_matrix)
model.save("models/PyGad.h5")

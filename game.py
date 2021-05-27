import time

from pettingzoo.mpe import simple_tag_v2
import click


def load_baseline_prey(n_predators, n_prey, n_obstacles):
    from prey import PreyBaseline
    return {f"agent_{i}": PreyBaseline(n_predators, n_prey, n_obstacles) for i in range(n_prey)}


def load_baseline_predators(n_predators, n_prey, n_obstacles):
    from predator import PredatorBaseline
    return {f"adversary_{i}": PredatorBaseline(n_predators, n_prey, n_obstacles) for i in range(n_predators)}


def load_advanced_prey():
    from dqn import PreyQLearning
    return {f"agent_0": PreyQLearning(n_predators=3, n_prey=1, n_obstacles=2, model_path="dqn_models/agent_0_model")}


def load_advanced_predators():
    from dqn import PredatorQLearning
    return {f"adversary_{i}": PredatorQLearning(n_predators=3, n_prey=1, n_obstacles=2,
                                                model_path=f"dqn_models/adversary_{i}_model") for i in range(3)}


def validate(n_predators, n_prey, n_obstacles, n_games, steps_per_game, prey, predator, seed, display_stats):
    if (prey == "advanced" or predator == "advanced") and not (n_predators == 3 and n_prey == 1 and n_obstacles == 2):
        print("To use an advanced agent the default values of predators, preys and obstacles should be used")
        exit()


@click.command()
@click.option("--n_predators", default=3, show_default=True, help="Number of predators")
@click.option("--n_prey", default=1, show_default=True, help="Number of prey")
@click.option("--n_obstacles", default=2, show_default=True, help="Number of obstacles")
@click.option("--n_games", default=10, show_default=True, help="Number of games")
@click.option("--steps_per_game", default=100, show_default=True, help="Number of steps per game")
@click.option("--prey", type=click.Choice(["baseline", "advanced"]), default="baseline", show_default=True,
              help="Type of prey")
@click.option("--predator", type=click.Choice(["baseline", "advanced"]), default="baseline", show_default=True,
              help="Type of predator")
@click.option("--seed", default=42, show_default=True, help="Random seed")
@click.option("--display_stats", default=True, show_default=True, help="Display stats after the simulation ends")
def run_game(n_predators, n_prey, n_obstacles, n_games, steps_per_game, prey, predator, seed, display_stats):
    validate(n_predators, n_prey, n_obstacles, n_games, steps_per_game, prey, predator, seed, display_stats)

    # setup environment
    env = simple_tag_v2.parallel_env(
        num_good=n_prey, num_adversaries=n_predators,
        num_obstacles=n_obstacles, max_cycles=steps_per_game)
    env.seed(seed=seed)

    # setup agents
    agents = {}

    if prey == "baseline":
        agents.update(load_baseline_prey(n_predators, n_prey, n_obstacles))
    elif prey == "advanced":
        agents.update(load_advanced_prey())
    else:
        raise Exception("Prey type is invalid")

    if predator == "baseline":
        agents.update(load_baseline_predators(n_predators, n_prey, n_obstacles))
    elif predator == "advanced":
        agents.update(load_advanced_predators())
    else:
        raise Exception("Predator type is invalid")

    # setup stats
    if display_stats:
        from monitor import Monitor
        monitor = Monitor(n_prey=n_prey, n_predators=n_predators, n_obstacles=n_obstacles,
                          steps_per_game=steps_per_game,
                          prey_names=[f"agent_{i}" for i in range(n_prey)],
                          predator_names=[f"adversary_{i}" for i in range(n_predators)])

    # game loop
    for game in range(n_games):
        state = env.reset()
        for _ in range(steps_per_game):
            env.render()

            actions = {}
            time.sleep(0.01)

            for name, agent in agents.items():
                agent.observe(state[name])
                actions[name] = agent.decide()

            state, rewards, done, _ = env.step(actions)

            if display_stats:
                monitor.log(state, rewards)

    env.close()

    if display_stats:
        monitor.stats()


if __name__ == "__main__":
    run_game()

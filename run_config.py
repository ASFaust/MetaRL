import sys
from utils import Config
from ga import create_population, evaluate, print_stats, create_new_population, save_best_actor

def main():
    config = Config.load(sys.argv[1])
    population = create_population(config)
    generation = 0
    while True:
        fitnesses = evaluate(population, config)
        print_stats(fitnesses)
        save_best_actor(population, fitnesses, generation, config)
        population = create_new_population(population, fitnesses, config)
        generation += 1


"""
    gen = 0
    while True:
        rewards = np.zeros(population_size)
        for i in range(evals_per_gen):
            print(f"Generation {gen + 1}, Evaluation {i + 1}/{evals_per_gen}")
            rewards += evaluate(population)
        rewards /= evals_per_gen

        sorted_indices = np.argsort(rewards)
        best_indices = sorted_indices[-n_survivors:]

        print(f"Generation {gen + 1}, Top 5 Rewards: {rewards[sorted_indices][-5:]}")
        print("Bottom 5 Rewards: ", rewards[sorted_indices][:5])

        #save the best actor
        print(f"saving best actor as best_params/learning_rule_gen_{gen:04d}.pt")
        params = get_params_by_index(population, best_indices[-1])
        torch.save(params, f"best_params/learning_rule_gen_{gen:04d}.pt")

        create_new_population(population, best_indices)
        gen += 1
"""

if __name__ == '__main__':
    main()

import sys
from utils import Config
from ga import create_population, evaluate, print_stats, create_new_population, save_best_actor

def main():
    config = Config.load(sys.argv[1])
    population = create_population(config)
    for generation in range(config.generations):
        fitnesses = evaluate(population, config)
        print_stats(fitnesses)
        save_best_actor(population, fitnesses, generation, config)
        population = create_new_population(population, fitnesses, config)

if __name__ == '__main__':
    main()

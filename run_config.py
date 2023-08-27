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

if __name__ == '__main__':
    main()

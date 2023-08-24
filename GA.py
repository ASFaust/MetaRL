import numpy as np
import torch

from GA_params import *

from LearningRule import LearningRule
from ActorNetwork import ActorNetwork
from DoublePendulum import DoublePendulum

def evaluate(params):
    print("evaluating")
    env = DoublePendulum(population_size)
    rewards = np.zeros(population_size)
    actors = create_actor_network(params)
    for i in range(num_steps):
        print("\rstep: {}/{}".format(i, num_steps), end="", flush=True)
        observations = env.get_state()
        actions = actors.forward(observations) * actor_force
        env.torque = actions.squeeze()
        env.step_rk4(sim_step_size)
        reward = env.get_reward()
        rewards += reward.cpu().numpy()
        actors.train(reward)
    print("\ndone")
    return rewards

def create_new_population(population, best_indices):
    print("creating new population")
    for i in range(population_size):
        print("\ractor: {}/{}".format(i, population_size), end="", flush=True)
        if i in best_indices:
            continue
        mutate_population(population, best_indices, i)
    print("\ndone")

def mutate_population(population, best_indices, i):
    parent = np.random.choice(best_indices)
    for network in population.keys():
        for param in population[network].keys():
            #sample from a normal distribution with mean 0 and std of mutation_strength * magnitude
            delta = torch.randn(population[network][param].shape[1:], device=device) * mutation_strength
            mask = (torch.rand(population[network][param].shape[1:], device=device) < mutation_probability).float()
            sparse_delta = delta * mask
            population[network][param][i] = population[network][param][parent]
            population[network][param][i] += sparse_delta

def main():
    population = LearningRule.get_random_params(population_size)

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

if __name__ == '__main__':
    main()

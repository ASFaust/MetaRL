import numpy as np
import torch
from LearningRule import LearningRule
from ActorNetwork import ActorNetwork
from util import count_parameters
from DoublePendulum import DoublePendulum

# Parameters
population_size = 10000
n_survivors = 200
generations = 1000
num_steps = 5000
evals_per_gen = 1
mutation_probability = 0.01  # 10% of the parameters will be mutated, adjust as necessary
mutation_strength = 0.1  # the standard deviation of the gaussian noise applied to the parameters
print("initializing environment")
print("environment initialized")

# Evaluation function
def evaluate(params):
    env = DoublePendulum(population_size)
    rewards = np.zeros(population_size)
    print("initializing actors")
    actors = ActorNetwork(
        [6, 8, 8, 1],
        LearningRule(params),
        weight_limit=4,
        sigma_limit=1,
        sigma_init=0.1,
        learning_rate=0.1,
        device='cuda',
        seed=None
    )
    print("actors initialized")
    print("evaluating actors", end="", flush=True)
    for i in range(num_steps):
        print(".", end="", flush=True)
        observations = env.get_state()
        actions = actors.forward(observations) * 5
        env.torque = actions.squeeze()
        env.step_rk4(0.1)
        reward = env.get_reward()
        rewards += reward.cpu().numpy()
        actors.train(reward)
    print("\ndone")
    return rewards

# Initialize environment and actors
population = LearningRule.get_random_params(population_size)

for gen in range(generations):
    rewards = np.zeros(population_size)
    for i in range(evals_per_gen):
        print(f"Generation {gen + 1}, Evaluation {i + 1}/{evals_per_gen}")
        rewards += evaluate(population)
    rewards /= evals_per_gen
    # make tuples of (reward, params)
    # but params are a dict of dict of pytorch tensors whose first dimension is the population size.

    # get me the indices of the highest rewards

    best_indices = np.argsort(rewards)[-n_survivors:]

    # print top 5 rewards
    print(f"Generation {gen + 1}, Top 5 Rewards: {rewards[best_indices][-5:]}")
    print("Bottom 5 Rewards: ", rewards[best_indices][:5])
    #now we need to make a new population
    print("creating new population")
    for i in range(population_size):
        if i in best_indices:
            continue  # don't overwrite the best ones
        parent = np.random.choice(best_indices)
        for network in population.keys():
            for param in population[network].keys():
                # Calculate the delta
                delta = torch.randn(population[network][param].shape[1:], device='cuda') * mutation_strength

                # Create a sparsity mask
                mask = (torch.rand(population[network][param].shape[1:], device='cuda') < mutation_probability).float()

                # Apply the mask to the delta
                sparse_delta = delta * mask

                # Modify the population tensor
                population[network][param][i] = population[network][param][parent]
                population[network][param][i] += sparse_delta
    if (gen + 1) % 2 == 0:
        # show the best one
        print("showing best one")
        best_params = {}
        for network in population.keys():
            best_params[network] = {}
            for param in population[network].keys():
                best_params[network][param] = population[network][param][best_indices[-1]:best_indices[-1] + 1]
        print("evaluating best actor")
        #make a new, non-vectorized environment
        env = DoublePendulum(1)
        actors = ActorNetwork(
            [6, 8, 8, 1],
            LearningRule(best_params),
            weight_limit=4,
            sigma_limit=1,
            sigma_init=0.1,
            learning_rate=0.1,
            device='cuda',
            seed=None
        )
        for i in range(num_steps):
            print(".", end="", flush=True)
            observations = env.get_state()
            actions = actors.forward(observations) * 5
            env.torque = actions.squeeze()
            env.step_rk4(0.1)
            reward = env.get_reward()
            env.render(0)
            actors.train(reward)
        print("\ndone")

env.close()

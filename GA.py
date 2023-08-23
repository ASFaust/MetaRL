import numpy as np
import gym.vector
import torch
from LearningRule import LearningRule
from ActorNetwork import ActorNetwork
from util import count_parameters

# Parameters
population_size = 100
n_survivors = 20
generations = 1000
num_steps = 200 #pendulum has a max of 200 steps
evals_per_gen = 1
mutation_probability = 0.01  # 10% of the parameters will be mutated, adjust as necessary
mutation_strength = 0.1  # the standard deviation of the gaussian noise applied to the parameters
print("initializing environment")
env = gym.vector.make("Pendulum-v1", num_envs=population_size)
print("environment initialized")

# Evaluation function
def evaluate(params):
    rewards = np.zeros(population_size)
    print("initializing actors")
    actors = ActorNetwork(
        [3, 8, 8, 1],
        LearningRule(params),
        weight_limit=4,
        sigma_limit=1,
        sigma_init=0.1,
        learning_rate=1.0,
        device='cuda',
        seed=None
    )
    print("actors initialized")
    print("evaluating actors", end="", flush=True)
    observations,_ = env.reset()
    for i in range(num_steps):
        print(".", end="", flush=True)
        x = torch.tensor(observations, device='cuda')
        actions = actors.forward(x) * 2
        actions = actions.cpu().numpy()
        #sample random actions
        #actions = np.random.randint(0,2,population_size)
        observations, rewards_, dones, truncates, info = env.step(actions)
        rewards += rewards_
        actors.train(torch.tensor(rewards_.astype(np.float32), device='cuda'))
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

    # show the best one
    print("showing best one")
    best_params = {}
    for network in population.keys():
        best_params[network] = {}
        for param in population[network].keys():
            best_params[network][param] = population[network][param][best_indices[-1]:best_indices[-1] + 1]

    best_actor = ActorNetwork(
        [3, 8, 8, 1],
        LearningRule(best_params),
        weight_limit=4,
        sigma_limit=1,
        sigma_init=0.1,
        learning_rate=1.0,
        device='cuda',
        seed=None
    )
    print("evaluating best actor")
    #make a new, non-vectorized environment
    env = gym.make("Pendulum-v1")
    observation,_ = env.reset()
    x = torch.tensor(observation, device='cuda')[None, :]
    for i in range(200):
        env.render()
        print("x shape")
        print(x.shape)
        action = best_actor.forward(x) * 2
        observation, reward, done, trunc, info = env.step(action.cpu().numpy())
        x = torch.tensor(observation, device='cuda')[0]

        best_actor.train(torch.tensor(reward.astype(np.float32), device='cuda').unsqueeze(0))


env.close()

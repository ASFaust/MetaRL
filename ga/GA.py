from environments import Environment
from networks import get_actor, LearningRule
import numpy as np
import torch
import yaml
import os

def create_population(config):
    return LearningRule.get_random_params(
        config.population_size,
        config.signal_dim,
        config.state_dim,
        config.node_info_dim,
        config.init_std,
        config.device)

def evaluate(params, config):
    sum_reward = np.ones(config.population_size) * 2**31
    step = 0
    n_steps = config.num_eval_steps * config.evals_per_gen
    env = Environment(config)
    for j in range(config.evals_per_gen):
        actor = get_actor(params, config)
        observation = env.reset()
        new_sum_reward = np.zeros(config.population_size)
        for i in range(config.num_eval_steps):
            step += 1
            print("\rstep: {}/{}".format(step, n_steps), end="", flush=True)
            action = actor.forward(observation)
            action = env.normalize_action(action)
            observation,reward = env.step(action)
            actor.train(reward)
            new_sum_reward += reward.cpu().numpy()
        sum_reward = np.minimum(sum_reward, new_sum_reward)
    #sum_reward /= config.evals_per_gen
    print("\ndone")
    indices = np.argsort(sum_reward)[::-1].copy()
    return indices, sum_reward

def print_stats(fitnesses):
    indices, sum_reward = fitnesses
    sr = sum_reward[indices]
    print("best reward: {}".format(sr[0]))
    print("worst reward: {}".format(sr[-1]))
    print("average reward: {}".format(np.mean(sum_reward)))
    print("median reward: {}".format(np.median(sum_reward)))

def select_inidviduals(population, indices):
    ret = {}
    for network in population.keys():
        ret[network] = {}
        for param in population[network].keys():
            param_type = type(population[network][param])
            if param_type == torch.Tensor:
                ret[network][param] = population[network][param][indices].clone()
            elif param_type == type(0):
                ret[network][param] = population[network][param]
            else:
                raise Exception("unknown param type: {}".format(param_type))
    return ret

def mutate_tensor(tensor, config):
    """
    Apply mutations to a tensor, starting after the survivors
    """
    # Create random mutation tensor for all elements, but only mutate after the first num_survivors elements
    delta = torch.randn_like(tensor, device=config.device) * config.mutation_strength
    mask = (torch.rand_like(tensor, device=config.device) < config.mutation_probability).float()
    sparse_delta = delta * mask
    tensor[config.num_survivors:] += sparse_delta[config.num_survivors:]
    return tensor

def create_new_population(population, fitnesses, config):
    indices, sum_reward = fitnesses
    #config.num_survivors holds the number of survivors
    #config.population_size holds the population size
    survivors = select_inidviduals(population, indices[:config.num_survivors])
    mutated = {}

    num_repeats = config.population_size // config.num_survivors
    remainder = config.population_size % config.num_survivors

    rep_ind = torch.arange(config.num_survivors).repeat_interleave(num_repeats).tolist()
    if remainder:
        rep_ind += torch.arange(remainder).tolist()

    for network in survivors.keys():
        mutated[network] = {}
        for param in survivors[network].keys():
            if type(survivors[network][param]) == type(0):
                mutated[network][param] = survivors[network][param]
                continue
            #survivors[network][param]: of dim (num_survivors, ...)
            #mutated[network][param]: of dim (population_size, ...)
            #what we want to do is to copy the survivors to the mutated population, by repeating the survivors until we reach the population size
            if remainder:
                mutated_tensor = survivors[network][param][rep_ind]
                mutated_tensor = torch.cat((mutated_tensor, survivors[network][param][:remainder]), dim=0).clone()
            else:
                mutated_tensor = survivors[network][param][rep_ind].clone()

            #now mutate the tensor, but not the first [config.num_survivors] elements. These are the survivors, they should not be mutated
            mutated_tensor = mutate_tensor(mutated_tensor, config)

            mutated[network][param] = mutated_tensor
    return mutated

def save_best_actor(population, fitnesses, generation, config):
    indices, sum_reward = fitnesses
    best_actor = select_inidviduals(population, [indices[0]])
    #first convert to cpu numpy
    for network in best_actor.keys():
        for param in best_actor[network].keys():
            param_type = type(best_actor[network][param])
            if param_type == torch.Tensor:
                best_actor[network][param] = best_actor[network][param].cpu().numpy().tolist()
            elif param_type == type(0):
                best_actor[network][param] = best_actor[network][param]
            else:
                raise Exception("unknown param type: {}".format(param_type))
    save_dict = {}
    save_dict["actor"] = best_actor
    save_dict["fitness"] = sum_reward[indices[0]].tolist()
    save_dict["config"] = config.to_dict()
    #save as yaml in results folder.
    #name is {timestamp}_{config_name}_gen{generation}_.yaml
    #config_name is the name of the config file, without the .yaml extension

    #first get the config name, which is config.path
    config_name = config.path.split("/")[-1].split(".")[0]
    timestamp = config.timestamp
    filename = "{}_{}_gen{:04d}.yaml".format(timestamp, config_name, generation)
    filepath = os.path.join("results", filename)
    with open(filepath, "w") as f:
        yaml.dump(save_dict, f, default_flow_style=True)

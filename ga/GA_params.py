from ActorNetwork import ActorNetwork
from LearningRule import LearningRule

# Global Parameters
population_size = 10000
n_survivors = 200
num_steps = 2000
evals_per_gen = 1

mutation_probability = 0.01
mutation_strength = 1.0
sim_step_size = 0.05

# ActorNetwork parameters
network_shape = [6, 16, 8, 1]
weight_limit = 4.0
sigma_limit = 1.0
sigma_init = 0.0
learning_rate = 1.0
device = 'cuda'
seed = None

#actor force
actor_force = 5 #[-1,1] * actor_force is the range of the force of the actor

def create_actor_network(params):
    return ActorNetwork(
        network_shape,
        LearningRule(params),
        weight_limit=weight_limit,
        sigma_limit=sigma_limit,
        sigma_init=sigma_init,
        learning_rate=learning_rate,
        device=device,
        seed=seed
    )

def get_params_by_index(population, index):
    best_params={}
    for network in population.keys():
        best_params[network] = {}
        for param in population[network].keys():
            best_params[network][param] = population[network][param][index:index+1]
    return best_params
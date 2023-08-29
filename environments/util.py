from .Pendulum import Pendulum
from .DoublePendulum import DoublePendulum
from .CartPole import CartPole
from .FoodWorld1 import FoodWorld1

def get_env(config):
    if config.env_name == 'DoublePendulum':
        return DoublePendulum(config.population_size, reward_type=config.reward_type, device=config.device)
    elif config.env_name == 'Pendulum':
        return Pendulum(config.population_size, reward_type=config.reward_type, device=config.device)
    elif config.env_name == 'CartPole':
        return CartPole(config.population_size, reward_type=config.reward_type, device=config.device)
    elif config.env_name == 'FoodWorld1':
        return FoodWorld1(config)
    else:
        raise NotImplementedError(f"Unknown environment: {config.env_name}")
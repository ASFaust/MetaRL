import torch
from environments import DoublePendulum, Pendulum

def get_env(config):
    if config.env == 'DoublePendulum':
        return DoublePendulum(config.population_size, device=config.device)
    elif config.env == 'Pendulum':
        return Pendulum(config.population_size, device=config.device)
    else:
        raise NotImplementedError


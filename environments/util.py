from .Pendulum import Pendulum
from .DoublePendulum import DoublePendulum

def get_env(config):
    if config.env_name == 'DoublePendulum':
        return DoublePendulum(config.population_size, device=config.device)
    elif config.env_name == 'Pendulum':
        return Pendulum(config.population_size, device=config.device)
    else:
        raise NotImplementedError
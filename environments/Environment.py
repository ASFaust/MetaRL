from .Pendulum import Pendulum
from .DoublePendulum import DoublePendulum
from .CartPole import CartPole
from .FoodWorld1 import FoodWorld1

"""
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
        
"""


class Environment:
    def __init__(self, config):
        if config.env_name == 'DoublePendulum':
            self.env = DoublePendulum(config)
        elif config.env_name == 'Pendulum':
            self.env = Pendulum(config)
        elif config.env_name == 'CartPole':
            self.env = CartPole(config)
        elif config.env_name == 'FoodWorld1':
            self.env = FoodWorld1(config)
        else:
            raise NotImplementedError(f"Unknown environment: {config.env_name}")
        if config.reward_type == 'value':
            self.reward = self.last_value
            self.reward_fn = self.env.get_value
        elif config.reward_type == 'delta':
            self.reward = 0
            self.last_value = self.env.get_value()
            self.reward_fn = self._delta_reward
        elif config.reward_type == 'penalized_delta':
            self.reward = 0
            self.last_value = self.env.get_value()
            self.reward_fn = self._penalized_delta_reward

    def step(self):
        self.env.step()

    def render(self, index):
        return self.env.render(index)

    def _delta_reward(self):
        value = self.env.get_value()
        delta = value - self.last_value
        self.last_value = value
        return delta

    def _penalized_delta_reward(self):
        # penalize negative rewards with a factor of 0.01
        value = self.env.get_value()
        delta = value - self.last_value
        self.last_value = value
        delta[delta < 0] *= 0.01
        return delta

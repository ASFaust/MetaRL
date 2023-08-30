from .Pendulum import Pendulum
from .DoublePendulum import DoublePendulum
from .CartPole import CartPole
from .FoodWorld1 import FoodWorld1
import torch

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
        self.env_assertions()


        self.last_value = self.env.get_value()
        if config.reward_type == 'value':
            self.reward = self.last_value
            self.reward_fn = self.env.get_value
        elif config.reward_type == 'delta':
            self.reward = 0
            self.reward_fn = self._delta_reward
        elif config.reward_type == 'penalized_delta':
            self.reward = 0
            self.reward_fn = self._penalized_delta_reward

        self.zero_action = torch.zeros((self.env.state.shape[0],self.env.action_dim), device=config.device)

        self.action_dim = self.env.action_dim
        self.observation_dim = self.env.observation_dim
        config.action_dim = self.action_dim
        config.observation_dim = self.observation_dim

    def step(self, action = None):
        if action is None:
            action = self.zero_action
        self.env.step(action)
        return self.env.get_observation(), self.reward_fn()

    def get_observation(self):
        return self.env.get_observation()

    def reset(self):
        self.env.reset()
        self.last_value = self.env.get_value()
        return self.env.get_observation()

    def normalize_action(self, action):
        #action is provided as [batch_dim,action_dim] with values in range range_high, range_low
        #normalize to the action space

        act = action + 1.0
        act = act / 2.0
        act = act * (self.env.action_space[:,1] - self.env.action_space[:,0])
        act = act + self.env.action_space[:,0]
        return act
    def random_action(self):
        #env has an action_space attribute
        #for example:
        #for FoodWorld1, action_space = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=self.device) * 0.001
        #sample from this action space. the first entries in the inner lists are the lower bounds, the second entries are the upper bounds
        return torch.rand((self.env.state.shape[0],self.env.action_dim), device=self.env.device) * (self.env.action_space[:,1] - self.env.action_space[:,0]) + self.env.action_space[:,0]

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

    def env_assertions(self):
        assert hasattr(self.env, 'step'), f"Environment {self.env} does not have a step method"
        assert hasattr(self.env, 'get_value'), f"Environment {self.env} does not have a get_value method"
        assert hasattr(self.env, 'render'), f"Environment {self.env} does not have a render method"
        #assert hasattr(self.env, 'random_action'), f"Environment {self.env} does not have a random_action method"
        assert hasattr(self.env, 'action_dim'), f"Environment {self.env} does not have an action_dim attribute"
        assert hasattr(self.env, 'observation_dim'), f"Environment {self.env} does not have a observation_dim attribute"
        assert hasattr(self.env, 'reset'), f"Environment {self.env} does not have a reset method"
        assert hasattr(self.env, 'get_observation'), f"Environment {self.env} does not have a get_observation method"

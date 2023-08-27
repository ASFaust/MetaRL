from .ActorNetwork import ActorNetwork
from .LearningRule import LearningRule
"""
    def __init__(
            self,
            layer_dims,
            learning_rule,
            weight_limit,
            weight_init,
            sigma_limit,
            sigma_init,
            learning_rate,
            device='cuda',
            seed=None):
"""

def get_actor(params,config):
    return ActorNetwork(
        layer_dims=config.network_shape,
        learning_rule=LearningRule(params),
        weight_limit=config.weight_limit,
        weight_init=config.weight_init,
        sigma_limit=config.sigma_limit,
        sigma_init=config.sigma_init,
        learning_rate=config.learning_rate,
        device=config.device,
        seed=config.seed,
        signal_dim=config.signal_dim
    )


from .ActorNetwork import ActorNetwork
from .LearningRule import LearningRule


def get_actor(params,config):
    return ActorNetwork(
        layer_dims=config.network_shape,
        learning_rule=LearningRule(params),
        config=config
    )

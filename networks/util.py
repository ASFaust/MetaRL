from .ActorNetwork import ActorNetwork
from .LearningRule import LearningRule


def get_actor(params,config):
    return ActorNetwork(
        learning_rule=LearningRule(params),
        config=config
    )

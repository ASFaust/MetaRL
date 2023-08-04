from LearningRuleNetworks import LearningRuleNetworks
from ActorNetwork import ActorNetwork

print(f"number of params: {LearningRuleNetworks.count_params()}")

params = LearningRuleNetworks.get_random_params(1)

lrn = LearningRuleNetworks(params)

#actor_network = ActorNetwork(3,3,lrn)



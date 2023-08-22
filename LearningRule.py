from LearningRuleNetworks.NodeNetwork import NodeNetwork
from LearningRuleNetworks.ConnectionNetwork import ConnectionNetwork
from LearningRuleNetworks.RewardNetwork import RewardNetwork

class LearningRule:
    def __init__(self,params):
        self.batch_dim = params["node_network"]["w1"].shape[0]
        self.node_network = NodeNetwork(params["node_network"])
        self.connection_network = ConnectionNetwork(params["connection_network"])
        self.reward_network = RewardNetwork(params["reward_network"])

    @staticmethod
    def get_random_params(batch_dim,device='cuda'):
        return {"node_network": NodeNetwork.get_random_params(batch_dim,device),
                "connection_network": ConnectionNetwork.get_random_params(batch_dim,device),
                "reward_network": RewardNetwork.get_random_params(batch_dim,device)}

from LearningRuleNetworks.NodeNetwork import NodeNetwork
from LearningRuleNetworks.ConnectionNetwork import ConnectionNetwork
from LearningRuleNetworks.RewardNetwork import RewardNetwork

class LearningRule:
    def __init__(self,params):
        self.batch_dim = params["node_network"]["w1"].shape[0]
        self.node_network = NodeNetwork(params["node_network"])
        self.connection_network = ConnectionNetwork(params["connection_network"])
        self.reward_network = RewardNetwork(params["reward_network"])
        self.params = params

    @staticmethod
    def get_random_params(batch_dim,device='cuda'):
        return {"node_network": NodeNetwork.get_random_params(batch_dim,device),
                "connection_network": ConnectionNetwork.get_random_params(batch_dim,device),
                "reward_network": RewardNetwork.get_random_params(batch_dim,device)}

    def save(self,path):
        #can torch save dictionaries? yes it can
        torch.save(self.params,path)

    @staticmethod
    def load(path,device='cuda'):
        params = torch.load(path) #will this return a dictionary?
        return LearningRule(params)
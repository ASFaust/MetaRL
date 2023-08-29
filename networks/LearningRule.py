from .LearningRuleNetwork import LearningRuleNetwork


class LearningRule:
    @staticmethod
    def get_random_params(batch_dim,
                          signal_dim,
                          state_dim,
                          node_info_dim,
                          init_std,
                          device='cuda'):
        return {"node_network": LearningRuleNetwork.get_random_params(
            batch_dim,
            input_dim=node_info_dim + signal_dim,
            output_dim=1 + 1 + signal_dim,  # 1 for weight, 1 for sigma, signal_dim for the signal
            state_dim=state_dim,
            std=init_std,
            device=device),
            "connection_network": LearningRuleNetwork.get_random_params(
                batch_dim,
                input_dim=node_info_dim * 2 + signal_dim + 1 + 1,
                # 2 for the two nodes, signal, weight, flag for input connection
                output_dim=1 + signal_dim,  # 1 for the weight, signal_dim for the signal
                state_dim=state_dim,
                std=init_std,
                device=device),
            "reward_network": LearningRuleNetwork.get_random_params(
                batch_dim,
                input_dim=node_info_dim + 1,  # reward and node_info are the inputs
                output_dim=signal_dim,  # signal_dim for the signal
                state_dim=state_dim,
                std=init_std,
                device=device)
        }

    def __init__(self, params):
        self.batch_dim = params["node_network"]["w_state"].shape[0]
        self.node_network = LearningRuleNetwork(params["node_network"])
        self.connection_network = LearningRuleNetwork(params["connection_network"])
        self.reward_network = LearningRuleNetwork(params["reward_network"])
        self.params = params

    def count_params(self):
        return self.node_network.count_params() + self.connection_network.count_params() + self.reward_network.count_params()


class Individuals:
    backward_info_dim = 5
    signal_dim = 1

    # connection_input has node info for the node that is sending the signal and the node that is receiving the signal
    # and the signal itself and the weight of the connection
    # and the flag for whether the connection is from hidden to hidden or from input to hidden
    connection_network_input_dim = node_info_dim * 2 + signal_dim + 1 + 1
    # it produces a weight change and a new signal
    connection_network_output_dim = signal_dim + 1

    connection_network_first_hidden_dim = signal_dim + 3


    # node network takes the node info and the signal
    node_network_input_dim = node_info_dim + signal_dim
    # and produces a new signal and a learning signal for the bias and the lambda
    node_network_output_dim = 2 + signal_dim

    #reward network takes the reward signal and the output value of the node
    reward_network_input_dim = 1 + 1

    #and it produces a learning signal for the node
    reward_network_output_dim = signal_dim

    connection_network_param_count = LearningRuleNetwork.count_params(
        connection_network_input_dim,
        connection_network_output_dim)

    node_network_param_count = LearningRuleNetwork.count_params(
        node_network_input_dim,
        node_network_output_dim)

    reward_network_param_count = LearningRuleNetwork.count_params(
        reward_network_input_dim,
        reward_network_output_dim)

    @staticmethod
    def count_params():
        return LearningRuleNetworks.connection_network_param_count + LearningRuleNetworks.node_network_param_count

    @staticmethod
    def get_random_params(batch_dim):
        return torch.randn(batch_dim, LearningRuleNetworks.count_params(), device='cuda')

    def __init__(self, params):
        # params is a tensor of shape (batch_dim, count_params)
        self.batch_dim = params.shape[0]
        self.connection_network = LearningRuleNetwork(
            LearningRuleNetworks.connection_network_input_dim,
            LearningRuleNetworks.connection_network_output_dim,
            self.get_connection_network_params(params),
        )
        self.node_network = LearningRuleNetwork(
            LearningRuleNetworks.node_network_input_dim,
            LearningRuleNetworks.node_network_output_dim,
            self.get_node_network_params(params),
        )
        self.reward_network = LearningRuleNetwork(
            LearningRuleNetworks.reward_network_input_dim,
            LearningRuleNetworks.reward_network_output_dim,
            self.get_reward_network_params(params),
        )


    def get_connection_network_params(self, params):
        return params[:, :LearningRuleNetworks.connection_network_param_count]

    def get_node_network_params(self, params):
        return params[:, LearningRuleNetworks.connection_network_param_count:]

    def get_reward_network_params(self, params):
        pos = LearningRuleNetworks.connection_network_param_count + LearningRuleNetworks.node_network_param_count
        return params[:, pos:]


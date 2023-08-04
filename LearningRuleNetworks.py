import torch
import numpy as np

#connection_input has shape (batch_dim, input_dim * output_dim, self.backward_info_dim * 2 + 1 + signal_dim)
#now we pass it through the connection network
#out = self.lrn.connection_network(connection_input)

# node input has shape (batch_dim, output_dim, (5 + signal_dim))
# out = self.lrn.node_network(node_input)
# out has shape (batch_dim, output_dim, (2 + signal_dim))

#so learning rule networks generally take a 3-dimensional input tensor and return a 3-dimensional output tensor
#they have different weights along the first dimension, but the same weights along the second dimension,
#and the third dimension is the input dimension of the network

class LearningRuleNetwork:
    def __init__(self, input_dim, output_dim, params):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = output_dim + 1
        self.init_network(params)

    def init_network(self, params):
        self.counter = 0

        def tensor_from_params(params, shape):
            num_elements = np.prod(shape)  # calculate total elements needed
            tensor = params[:, self.counter:self.counter + num_elements]
            tensor = tensor.reshape(-1, *shape)
            self.counter += num_elements
            return tensor

        self.w1 = tensor_from_params(params, (self.input_dim, self.hidden_dim))
        self.b1 = tensor_from_params(params, (self.hidden_dim,))
        self.w2 = tensor_from_params(params, (self.hidden_dim, self.hidden_dim))

    def forward(self,x):
        #x has shape (batch_dim, some_dim, output_dim)
        #w1 has shape (batch_dim, input_dim, output_dim)
        #b1 has shape (batch_dim, output_dim)
        #w2 has shape (batch_dim, output_dim, output_dim)
        #b2 has shape (batch_dim, output_dim)

        #w1[i] is the weight matrix for the inputs x[i]
        #x[i] has shape (some_dim, output_dim)
        #w1[i] should be matmulled with every x[i][j] for j in range(some_dim)

        h1 = torch.tanh(self.mm(x, self.w1) + self.b1)
        h2_1 = torch.tanh(self.mm(h1, self.w2_1) + self.b2_1)
        h2_2 = torch.tanh(self.mm(h1, self.w2_2) + self.b2_2)
        h2 = h2_1 * h2_2
        h3 = torch.tanh(self.mm(h2, self.w_3) + self.b_3)
        return h3

    def mm(self, x, w):
        #wt = w[:, None, :, :]
        #xt = x[:, :, None, :]
        ##result = wt * xt
        #return result.sum(dim=2)
        return torch.einsum('biod,bjd->bij', (w, x))

    @staticmethod
    def count_params(input_dim, output_dim):
        #we have a weight matrix of shape (input_dim, output_dim) and another one of shape (output_dim, output_dim)
        #and a bias vector of shape (output_dim) and another one of shape (output_dim)
        return output_dim * (input_dim + (output_dim + 1) * 2 + 3 + output_dim)

#actor or learningrulenetworks or
class LearningRuleNetworks:
    signal_dim = 1
    node_info_dim = 5
    #connection_input has node info for the node that is sending the signal and the node that is receiving the signal
    #and the signal itself and the weight of the connection
    connection_network_input_dim = node_info_dim * 2 + 1 + signal_dim
    #it produces a weight change and a new signal
    connection_network_output_dim = signal_dim + 1
    #node network takes the node info and the signal
    node_network_input_dim = node_info_dim + signal_dim
    #and produces a new signal and a learning signal for the bias and the lambda
    node_network_output_dim = 2 + signal_dim

    connection_network_param_count = LearningRuleNetwork.count_params(
        connection_network_input_dim,
        connection_network_output_dim)

    node_network_param_count = LearningRuleNetwork.count_params(
        node_network_input_dim,
        node_network_output_dim)

    @staticmethod
    def count_params():
        return LearningRuleNetworks.connection_network_param_count + LearningRuleNetworks.node_network_param_count

    @staticmethod
    def get_random_params(batch_dim):
        return torch.randn(batch_dim, LearningRuleNetworks.count_params(),device='cuda')

    def __init__(self, params):
        #params is a tensor of shape (batch_dim, count_params)
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

    def get_connection_network_params(self, params):
        return params[:, :LearningRuleNetworks.connection_network_param_count]

    def get_node_network_params(self, params):
        return params[:, LearningRuleNetworks.connection_network_param_count:]

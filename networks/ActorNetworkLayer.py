import torch


class ActorNetworkLayer:
    def __init__(
            self,
            input_dim,
            output_dim,
            learning_rule,
            previous_layer,
            seed,
            config
    ):
        self.learning_rule = learning_rule
        self.batch_dim = learning_rule.batch_dim
        self.learning_rate = config.learning_rate

        self.weight_limit = config.weight_limit
        self.sigma_limit = config.sigma_limit

        self.device = config.device

        if seed is not None:
            torch.manual_seed(seed)
        weight_range = config.weight_init * config.weight_limit
        self.weights = torch.randn((self.batch_dim, input_dim, output_dim), device=self.device) * weight_range
        self.biases = torch.randn(self.batch_dim, output_dim, device=self.device) * weight_range
        self.sigmas = torch.ones(self.batch_dim, output_dim, device=self.device) * config.sigma_init

        self.out_dim = output_dim
        self.in_dim = input_dim

        # needed for backward pass:
        self.previous_layer = previous_layer

        self.sum = None
        self.samples = None
        self.output = None

        self.node_network_state = None
        self.connection_network_state = None

        self.node_info_dim = config.node_info_dim
        self.signal_dim = config.signal_dim

        self.reset()

    def reset(self):
        self.node_network_state = self.learning_rule.node_network.get_zero_state(self.out_dim)
        self.connection_network_state = self.learning_rule.connection_network.get_zero_state(self.out_dim * self.in_dim)

    def forward(self, x):
        # x: (batch_dim, input_dim) float tensor
        # self.weights has shape (batch_dim, input_dim, output_dim)

        self.sum = torch.einsum('bi,bio->bo', x, self.weights)
        # self.sum /= self.in_dim

        self.sum += self.biases
        self.samples = torch.normal(self.sum, self.sigmas)
        self.output = torch.tanh(self.samples)  # tanh or swish?
        return self.output

    def get_node_info(self):
        # returns a tensor with shape (batch_dim, output_dim, 5) by stacking the tensors
        node_info = torch.stack([self.sum, self.samples, self.output, self.biases, self.sigmas], dim=2)
        return node_info

    def backward(self, signal, last_input=None):
        # signal: (batch_dim, output_dim, signal_dim)
        # signal is the learning signal from the next layer
        # it is already accumulated over the connections.
        # so first we pass the signal through the neuron network for every neuron.
        # then we pass the signal through the connection network for every connection, and accumulate the signals
        # to get a signal of shape (batch_dim, input_dim, grad_dim)

        # along the batch dim, we need to use different entries of self.learning_rules:
        # for batch dim = 0, we use self.learning_rules[0] etc.

        node_input = torch.cat([self.get_node_info(), signal], dim=2)

        # should have shape (batch_dim,output_dim, (5 + signal_dim))
        # one set of weights of the node network is used for [0,:,:], another for [1,:,:] etc.
        # so the input dimensionality of the node network is 5 + signal_dim, not too bad.
        # if we use moving average + variance, it is 3 * (5 + signal_dim),
        # which is quite large for a genetic algorithm to optimize

        out, self.node_network_state = self.learning_rule.node_network.forward(node_input, self.node_network_state)

        # out contains a new signal that is passed to each connection
        # and out also contains training signals for biases and lambdas
        # so out has shape (batch_dim, output_dim, (2 + signal_dim))
        # we need to split this into three parts:
        # signal for the next layer, signal for the biases, signal for the lambdas
        # the signal for the next layer is the first signal_dim entries
        signal_connections = out[:, :, 0:self.signal_dim]
        # the signal for the biases is the next entry
        signal_bias = out[:, :, self.signal_dim]
        # has shape (batch_dim, output_dim)
        # the signal for the sigmas is the last entry
        signal_sigma = out[:, :, self.signal_dim + 1]

        # now alter sigmas and biases

        self.sigmas = self.sigmas + signal_sigma * self.learning_rate
        # clip sigmas to [0, sigma_limit]
        self.sigmas = torch.clamp(self.sigmas, 0.0, self.sigma_limit)

        self.biases = self.biases + signal_bias * self.learning_rate
        # clip biases to [-weight_limit, weight_limit]
        self.biases = torch.clamp(self.biases, -self.weight_limit, self.weight_limit)

        node_info = self.get_node_info()[:, None, :, :]
        node_info = node_info.expand(-1, self.in_dim, -1, -1)

        if last_input is None:

            flag_previous_layer = torch.zeros(self.batch_dim, self.in_dim, self.out_dim, 1, device=self.device)
            prev_node_info = self.previous_layer.get_node_info()[:, :, None, :]
            prev_node_info = prev_node_info.expand(-1, -1, self.out_dim, -1)

        else:
            flag_previous_layer = torch.ones(self.batch_dim, self.in_dim, self.out_dim, 1, device=self.device)
            prev_node_info = torch.zeros(self.batch_dim, self.in_dim, self.out_dim, self.node_info_dim,
                                         device=self.device)
            # last_input has shape (batch_dim, input_dim)
            # output of last layer gets filled with last_input.
            # can be dangerous if last_input magnitude is significantly
            # larger than [-1,1]
            prev_node_info[:, :, :, 2] = last_input[:, :, None]

        signal_connections = signal_connections[:, None, :, :]
        signal_connections = signal_connections.expand(-1, self.in_dim, -1, -1)

        weight_info = self.weights[:, :, :, None]

        connection_input = torch.cat(
            [node_info, prev_node_info, signal_connections, weight_info, flag_previous_layer], dim=3)
        # connection_input has shape (batch_dim, input_dim, output_dim, self.node_info_dim * 2 + 1 + signal_dim)
        # reshape it to (batch_dim, input_dim * output_dim, self.node_info_dim * 2 + 1 + signal_dim)
        connection_input = connection_input.reshape(self.batch_dim, self.in_dim * self.out_dim, -1)
        # now we pass it through the connection network
        out, self.connection_network_state = self.learning_rule.connection_network.forward(connection_input,
                                                                                           self.connection_network_state)
        # out has shape (batch_dim, output_dim * input_dim, signal_dim + 1),
        # where + 1 is finally the weight update signal
        # reshape it to (batch_dim, output_dim, input_dim, signal_dim + 1)
        out = out.reshape(self.batch_dim, self.in_dim, self.out_dim, -1)
        # so we split it into two parts

        # the first part is the signal for the next layer
        signal_next_layer = out[:, :, :, 0:self.signal_dim]
        # has shape (batch_dim, output_dim, input_dim, signal_dim)
        # we need to take the mean over the output dimension to get a signal of shape (batch_dim, input_dim, signal_dim)
        # this is the signal that we pass to the next layer
        signal_next_layer = torch.mean(signal_next_layer, dim=2)  # todo: maybe more, different aggregation functions?

        # the second part is the weight update signal
        signal_weight_update = out[:, :, :, self.signal_dim]
        # has shape (batch_dim, output_dim, input_dim), same as self.weights
        self.weights = self.weights + signal_weight_update * self.learning_rate
        # clip weights
        self.weights = torch.clamp(self.weights, -self.weight_limit, self.weight_limit)

        return signal_next_layer

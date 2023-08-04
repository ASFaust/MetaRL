import torch

class ActorNetworkLayer:
    def __init__(self, input_dim, output_dim, lrn, previous_layer):
        #lrn: learning rule networks
        self.learning_rule_networks = lrn
        self.batch_dim = lrn.batch_dim

        self.weights = torch.rand(self.batch_dim, input_dim, output_dim, device = 'cuda') * 2 - 1
        self.biases = torch.zeros(self.batch_dim, output_dim, device = 'cuda')
        self.lambdas = torch.ones(self.batch_dim, output_dim, device = 'cuda') * 0.2 #start with a low lambda

        self.out_dim = output_dim
        self.in_dim = input_dim

        #needed for backward pass:
        self.previous_layer = previous_layer

        self.backward_info_dim = 5
        self.signal_dim = 1

    def forward(self, x):
        #x: (batch_dim, input_dim) float tensor
        output = torch.matmul(x.unsqueeze(1), self.weights).squeeze(1) + self.biases
        self.sigmoid = torch.sigmoid(output / self.out_dim)
        # sample from {0,1} with p(1) = sigmoid, p(0) = 1 - sigmoid:
        self.samples = torch.bernoulli(self.sigmoid)
        self.output = self.samples * self.lambdas + (1.0 - self.samples) * self.sigmoid
        #output: (batch_dim, output_dim) float tensor
        return self.output

    def get_backward_info(self):
        #returns a tensor with shape (batch_dim, output_dim, 5) by stacking the tensors
        return torch.stack([self.output, self.sigmoid, self.samples, self.lambdas, self.biases], dim = 2)

    def backward(self, signal):
        #signal: (batch_dim, output_dim, signal_dim)
        #signal is the learning signal from the next layer
        #it is already accumulated over the connections.
        #so first we pass the signal through the neuron network for every neuron.
        #then we pass the signal through the connection network for every connection, and accumulate the signals
        #to get a signal of shape (batch_dim, input_dim, grad_dim)

        #along the batch dim, we need to use different entries of self.learning_rules:
        #for batch dim = 0, we use self.learning_rules[0] etc.

        #can we vectorize this?

        #self.learning_rules[0].node_network(self.output, signal)

        node_input = torch.cat([self.get_backward_info(),signal], dim = 2)

        #should have shape (batch_dim,output_dim, (5 + signal_dim))

        #one set of weights of the node network is used for [0,:,:], another for [1,:,:] etc.
        #so the input dimensionality of the node network is 5 + signal_dim, not too bad.
        # if we use moving average + variance, it is 3 * (5 + signal_dim),
        # which is quite large for a genetic algorithm to optimize

        out = self.lrn.node_network(node_input)

        #out contains a new signal that is passed to each connection
        #and out also contains training signals for biases and lambdas
        #so out has shape (batch_dim, output_dim, (2 + signal_dim))
        #we need to split this into three parts:
        #signal for the next layer, signal for the biases, signal for the lambdas
        #the signal for the next layer is the first signal_dim entries
        signal_connections = out[:,:,0:self.signal_dim]
        #the signal for the biases is the next entry
        signal_bias = out[:,:,self.signal_dim]
        #has shape (batch_dim, output_dim)
        #the signal for the lambdas is the last entry
        signal_lambda = out[:,:,self.signal_dim + 1]

        #now alter lambda and bias
        self.lambdas = self.lambdas + signal_lambda
        self.biases = self.biases + signal_bias

        #clip lambdas to [0,1]
        self.lambdas = torch.clamp(self.lambdas, 0.0, 1.0)

        #clip biases to [-1,1]
        self.biases = torch.clamp(self.biases, -1.0, 1.0)

        backward_info = self.get_backward_info()[:, None, :, :]
        backward_info = backward_info.expand(-1, self.in_dim, -1, -1)

        prev_backward_info = self.previous_layer.get_backward_info()[:, :, None, :]
        prev_backward_info = prev_backward_info.expand(-1, -1, self.out_dim, -1)

        signal_connections = signal_connections[:, None, :, :]
        signal_connections = signal_connections.expand(-1, self.in_dim, -1, -1)

        weight_info = self.weights[:, :, :, None]

        connection_input = torch.cat([backward_info, prev_backward_info, signal_connections, weight_info], dim=3)
        #connection_input has shape (batch_dim, input_dim, output_dim, self.backward_info_dim * 2 + 1 + signal_dim)
        #reshape it to (batch_dim, input_dim * output_dim, self.backward_info_dim * 2 + 1 + signal_dim)
        connection_input = connection_input.reshape(self.batch_dim, self.in_dim * self.out_dim, -1)
        #now we pass it through the connection network
        out = self.lrn.connection_network(connection_input)
        #out has shape (batch_dim, output_dim * input_dim, signal_dim + 1), where + 1 is finally the weight update signal
        #reshape it to (batch_dim, output_dim, input_dim, signal_dim + 1)
        out = out.reshape(self.batch_dim, self.out_dim, self.in_dim, -1)
        #so we split it into two parts

        #the first part is the signal for the next layer
        signal_next_layer = out[:,:,:,0:self.signal_dim]
        #has shape (batch_dim, output_dim, input_dim, signal_dim)
        #we need to take the mean over the output dimension to get a signal of shape (batch_dim, input_dim, signal_dim)
        #this is the signal that we pass to the next layer
        signal_next_layer = torch.mean(signal_next_layer, dim=1) #todo: maybe more, different aggregation functions?

        #the second part is the weight update signal
        signal_weight_update = out[:,:,:,self.signal_dim]
        #has shape (batch_dim, output_dim, input_dim), same as self.weights
        self.weights = self.weights + signal_weight_update
        #clip weights to [-2,2]
        self.weights = torch.clamp(self.weights, -2.0, 2.0)

        return signal_next_layer

class ActorNetworkInputLayer:
    def __init__(self):
        pass
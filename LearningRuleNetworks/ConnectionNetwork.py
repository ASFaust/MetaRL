import torch
import numpy as np

# connection_input has shape (batch_dim, input_dim * output_dim, self.backward_info_dim * 2 + 1 + signal_dim)
# now we pass it through the connection network
# out = self.lrn.connection_network(connection_input)

# node input has shape (batch_dim, output_dim, (5 + signal_dim))
# out = self.lrn.node_network(node_input)
# out has shape (batch_dim, output_dim, (2 + signal_dim))

# so learning rule networks generally take a 3-dimensional input tensor and return a 3-dimensional output tensor
# they have different weights along the first dimension, but the same weights along the second dimension,
# and the third dimension is the input dimension of the network

class ConnectionNetwork:
    #input dim is 13
    #output dim is 2
    #state dim is 8

    def __init__(self, params):
        self.batch_dim = params["w1"].shape[0]

        # the input embedding layer:(gets mapped onto 5 values: the first two outputs are multiplied)
        self.w1 = params["w1"]  # tensor_from_params(params, (21, 2)) #52 values
        self.b1 = params["b1"]  # tensor_from_params(params, (2,)) #4 values

        # the recurrent layer: (gets mapped onto 3 values: the first two outputs are multiplied)
        self.w2 = params["w2"]  # tensor_from_params(params, (21, 8)) #recurrent connections, 16 values
        self.b2 = params["b2"]  # tensor_from_params(params, (21,)) #2 values

    @staticmethod
    def get_random_params(batch_dim, device='cuda'):
        w1 = torch.randn(batch_dim, 21, 2, device=device) * 0.01
        b1 = torch.randn(batch_dim, 2, device=device) * 0.01
        w2 = torch.randn(batch_dim, 21, 8, device=device) * 0.01
        b2 = torch.randn(batch_dim, 8, device=device) * 0.01
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def get_zero_state(self, some_dim):
        return torch.zeros(self.batch_dim, some_dim, 8, device=self.w1.device)

    def forward(self, x, state):
        # x has shape (batch_dim, some_dim, input_dim)
        # we want to return a tensor of shape (batch_dim, some_dim, output_dim)
        # first we map
        input_h2 = torch.cat((state, x), dim=2)  # we have 8 hidden states, so this is a 5 + 8 = 13 dimensional vector
        new_state = torch.tanh(self.mm(input_h2, self.w2, self.b2))
        # new_state has shape (batch_dim, some_dim, 8)
        input_h1 = torch.cat((new_state, x), dim=2)  # we have 8 hidden states, so this is a 5 + 8 = 13 dimensional vector
        h1 = torch.tanh(self.mm(input_h1, self.w1, self.b1))
        return h1, new_state

    def mm(self, x, w, b):
        matmul = torch.einsum('bsi,bio->bso', (x, w))
        return matmul + b.unsqueeze(1)


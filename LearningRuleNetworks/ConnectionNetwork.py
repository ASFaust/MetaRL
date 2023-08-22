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
    param_count = 92

    def __init__(self, params):
        self.batch_dim = params["w1"].shape[0]

        # the input embedding layer:(gets mapped onto 5 values: the first two outputs are multiplied)
        self.w1 = params["w1"]  # tensor_from_params(params, (13, 4)) #52 values
        self.b1 = params["b1"]  # tensor_from_params(params, (4,)) #4 values

        # the recurrent layer: (gets mapped onto 3 values: the first two outputs are multiplied)
        self.w2 = params["w2"]  # tensor_from_params(params, (8, 2)) #recurrent connections, 16 values
        self.b2 = params["b2"]  # tensor_from_params(params, (2,)) #2 values

        # the output layer: (2 outputs: weight change and new signal)
        self.w3 = params["w3"]  # tensor_from_params(params, (8, 2))
        self.b3 = params["b3"]  # tensor_from_params(params, (2,)) #tensor_from_params(params, (2,))

    @staticmethod
    def get_random_params(batch_dim, device='cuda'):
        w1 = torch.randn(batch_dim, 13, 4, device=device)
        b1 = torch.randn(batch_dim, 4, device=device) * 0.1
        w2 = torch.randn(batch_dim, 8, 2, device=device)
        b2 = torch.randn(batch_dim, 2, device=device) * 0.1
        w3 = torch.randn(batch_dim, 8, 2, device=device)
        b3 = torch.randn(batch_dim, 2, device=device) * 0.1
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

    def get_zero_state(self, some_dim):
        return torch.zeros(self.batch_dim, some_dim, 2, device=self.w1.device)

    def forward(self, x, state):
        # x has shape (batch_dim, some_dim, input_dim)
        # we want to return a tensor of shape (batch_dim, some_dim, output_dim)
        # first we map
        h1 = self.mm(x, self.w1) + self.b1
        h1t = h1[:, :, 0] * h1[:, :, 1]
        h1t = h1t.unsqueeze(2)
        h1 = torch.cat((h1, h1t), dim=2)
        h1 = torch.tanh(h1)

        # now i want to enable recurrent connections
        input_h2 = torch.cat((state, h1), dim=2)  # we have 2 hidden states, so this is a 5 + 3 = 7 dimensional vector
        h2 = self.mm(input_h2, self.w2) + self.b2
        h2t = h2[:, :, 0] * h2[:, :, 1]
        h2t = h2t.unsqueeze(2)
        # this allows hebbian learning and maybe even spike timing dependent plasticity in some parametrization
        h2 = torch.cat((h2, h2t), dim=2)
        h2 = torch.tanh(h2)

        new_state = h2  # this is 3 dimensional
        input_h3 = torch.cat((h2, h1), dim=2)  # 8 dimensional
        h3 = self.mm(input_h3, self.w3) + self.b3  # signal_dim + weight_update_dim = 2
        h3 = torch.tanh(h3)
        return h3, new_state

    def mm(self, x, w):
        return torch.einsum('bsi,bio->bso', (x, w))

import torch
import numpy as np


class RewardNetwork:
    param_count = 14

    def __init__(self, params):
        self.batch_dim = params["w1"].shape[0]

        # the 2 recurrent neurons:
        self.w1 = params["w1"]  # (batch_dim, 2 + 2, 2), so 8 weights
        self.b1 = params["b1"]  # (batch_dim, 2), so 2 biases

        # the output neuron:
        self.w2 = params["w2"]  # (batch_dim, 2 + 2, 1), so 4 weights
        self.b2 = params["b2"]  # (batch_dim, 1), so 1 bias: the learning signal

    @staticmethod
    def get_random_params(batch_dim, device='cuda'):
        w1 = torch.randn(batch_dim, 2 + 2, 2, device=device)
        b1 = torch.randn(batch_dim, 2, device=device) * 0.1
        w2 = torch.randn(batch_dim, 2 + 2, 1, device=device)
        b2 = torch.randn(batch_dim, 1, device=device) * 0.1
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def get_zero_state(self, some_dim):
        return torch.zeros(self.batch_dim, some_dim, 2, device=self.w1.device)

    def forward(self, x, state):
        # x has shape (batch_dim, some_dim, input_dim)
        # we want to return a tensor of shape (batch_dim, some_dim, output_dim)
        # first we map
        input_h1 = torch.cat((state, x), dim=2)  # 8 dimensional

        h1 = self.mm(input_h1, self.w1, self.b1)
         #+ self.b1
        h1 = torch.tanh(h1)

        input_h2 = torch.cat((state, h1), dim=2)  # 8 dimensional

        h2 = self.mm(input_h2, self.w2, self.b2)
        h2 = torch.tanh(h2)

        new_state = h1  # this is 3 dimensional

        return h2, new_state

    def mm(self, x, w, b):
        matmul = torch.einsum('bsi,bio->bso', (x, w))
        return matmul + b.unsqueeze(1)
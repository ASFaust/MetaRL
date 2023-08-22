import torch
import numpy as np

class NodeNetwork
    param_count = 66

    def __init__(self,params):
        self.batch_dim = params["w1"].shape[0]

        #the input embedding layer:(gets mapped onto 5 values: the first two outputs are multiplied)
        self.w1 = params["w1"] #6, 4)) #24 values
        self.b1 = params["b1"] #tensor_from_params(params, (4,)) #4 values

        self.w2 = params["w2"] #tensor_from_params(params, (6, 2)) #12 values
        self.b2 = params["b2"] #tensor_from_params(params, (2,)) #2 recurrent neurons

        self.w3 = params["w3"] #tensor_from_params(params, (6, 3)) #21 values: the new recurrent values and the h1 from w1,b1 x
        self.b3 = params["b3"] #tensor_from_params(params, (3,)) #3 values: signal, batch change, sigma change

        #total parameter count: 24 + 4 + 12 + 2 + 21 + 3 = 66

    @staticmethod
    def get_random_params(batch_dim,device='cuda'):
        ret_dict = {}
        ret_dict["w1"] = torch.randn(batch_dim, 6, 4, device=device)
        ret_dict["b1"] = torch.randn(batch_dim, 4, device=device) * 0.1
        ret_dict["w2"] = torch.randn(batch_dim, 6, 2, device=device)
        ret_dict["b2"] = torch.randn(batch_dim, 2, device=device) * 0.1
        ret_dict["w3"] = torch.randn(batch_dim, 6, 3, device=device)
        ret_dict["b3"] = torch.randn(batch_dim, 3, device=device) * 0.1
        return ret_dict

    def get_zero_state(self, some_dim):
        return torch.zeros(self.batch_dim, some_dim, 2, device=self.w1.device)

    def forward(self, x, state):
        # x has shape (batch_dim, some_dim, input_dim)
        # we want to return a tensor of shape (batch_dim, some_dim, output_dim)
        # first we map
        h1 = self.mm(x, self.w1) + self.b1
        h1 = torch.tanh(h1)

        input_h2 = torch.cat((state, h1), dim=2) #6 dimensional recurrent input
        h2 = self.mm(input_h2, self.w2) + self.b2 #
        h2 = torch.tanh(h2)

        input_h3 = torch.cat((h2, h1), dim=2) #6 dimensional recurrent input
        h3 = self.mm(input_h3, self.w3) + self.b3 #
        h3 = torch.tanh(h3)

        new_state = h2

        return h3, new_state

    def mm(self, x, w):
        return torch.einsum('bsi,bio->bso', (x, w))




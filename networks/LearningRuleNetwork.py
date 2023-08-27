import torch
import numpy as np

class LearningRuleNetwork:
    @staticmethod
    def get_random_params(batch_dim, input_dim, output_dim, state_dim, std, device='cuda'):
        w_state = torch.randn(batch_dim, input_dim + state_dim, output_dim, device=device) * std
        b_state = torch.randn(batch_dim, state_dim, device=device) * std
        w_output = torch.randn(batch_dim, input_dim + state_dim, output_dim, device=device) * std
        b_output = torch.randn(batch_dim, output_dim, device=device) * std
        return {"w_state": w_state, "b_state": b_state, "w_output": w_output, "b_output": b_output}

    def __init__(self, params):
        self.batch_dim = params["w_state"].shape[0]
        self.state_dim = params["w_state"].shape[2]
        self.input_dim = params["w_state"].shape[1] - self.state_dim
        self.output_dim = params["w_output"].shape[2]
        self.w_state = params["w_state"]  # (batch_dim, input_dim + state_dim, state_dim),
        self.b_state = params["b_state"]  # (batch_dim, state_dim),
        self.w_output = params["w_output"]  # (batch_dim, input_dim + state_dim, output_dim),
        self.b_output = params["b_output"]  # (batch_dim, output_dim),

    def get_zero_state(self, some_dim):
        return torch.zeros(self.batch_dim, some_dim, self.state_dim, device=self.w1.device)

    def forward(self, x, state):
        input_state = torch.cat((state, x), dim=2)
        new_state = torch.tanh(self.mm(input_state, self.w2, self.b2))
        input_out = torch.cat((new_state, x), dim=2)
        out = self.mm(input_out, self.w1, self.b1)
        return out, new_state

    def mm(self, x, w, b):
        matmul = torch.einsum('bsi,bio->bso', (x, w))
        return matmul + b.unsqueeze(1)

    def count_params(self):
        return np.prod(self.w_state.shape) + np.prod(self.b_state.shape) + np.prod(self.w_output.shape) + np.prod(self.b_output.shape)



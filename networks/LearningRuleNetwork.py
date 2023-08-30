import torch
import numpy as np


class LearningRuleNetwork:
    @staticmethod
    def get_random_params(batch_dim, input_dim, output_dim, state_dim, std, device='cuda'):
        w_output = torch.randn(batch_dim, input_dim + state_dim, output_dim, device=device) * std
        b_output = torch.randn(batch_dim, output_dim, device=device) * std

        if state_dim > 0:
            w_state = torch.randn(batch_dim, input_dim + state_dim, state_dim, device=device) * std
            b_state = torch.randn(batch_dim, state_dim, device=device) * std
            return {"w_state": w_state, "b_state": b_state, "w_output": w_output, "b_output": b_output,
                    "state_dim": state_dim}
        else:
            return {"w_output": w_output, "b_output": b_output, "state_dim": state_dim}

    def __init__(self, params):
        self.batch_dim = params["w_output"].shape[0]
        self.state_dim = params["state_dim"]
        self.input_dim = params["w_output"].shape[1] - self.state_dim
        self.output_dim = params["w_output"].shape[2]

        if self.state_dim > 0:
            self.w_state = params["w_state"]  # (batch_dim, input_dim + state_dim, state_dim),
            self.b_state = params["b_state"]  # (batch_dim, state_dim),
            self.forward = self.forward_with_state
            self.get_zero_state = self._get_zero_state
        else:
            self.forward = self.forward_without_state
            self.get_zero_state = lambda x: None
        self.w_output = params["w_output"]  # (batch_dim, input_dim + state_dim, output_dim),
        self.b_output = params["b_output"]  # (batch_dim, output_dim),

    def _get_zero_state(self, some_dim):
        return torch.zeros(self.batch_dim, some_dim, self.state_dim, device=self.w_state.device)

    def forward_with_state(self, x, state):
        input_state = torch.cat((state, x), dim=2)
        new_state = torch.tanh(LearningRuleNetwork.mm(input_state, self.w_state, self.b_state))
        input_out = torch.cat((new_state, x), dim=2)
        out = LearningRuleNetwork.mm(input_out, self.w_output, self.b_output)
        out = torch.tanh(out)
        return out, new_state

    def forward_without_state(self, x, _):
        out = LearningRuleNetwork.mm(x, self.w_output, self.b_output)
        out = torch.tanh(out)
        return out, None

    @staticmethod
    def mm(x, w, b):
        matmul = torch.einsum('bsi,bio->bso', (x, w))
        return matmul + b.unsqueeze(1)

    # dont count batch dim in count_params. prod shape[1:]!
    def count_params(self):
        return np.prod(self.w_state.shape[1:]) + np.prod(self.b_state.shape[1:]) + np.prod(
            self.w_output.shape[1:]) + np.prod(self.b_output.shape[1:])

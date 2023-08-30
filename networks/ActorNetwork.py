from .ActorNetworkLayer import ActorNetworkLayer
import torch


class ActorNetwork:
    def __init__(
            self,
            learning_rule,
            config):
        # if seed is none, the parameters are initialized randomly, but the same for every batch
        # if seed is not none, the parameters are initialized randomly, still the same for every batch,
        # but according to the seed
        self.input_dim = config.network_shape[0]
        self.output_dim = config.network_shape[-1]

        self.action_location = config.action_location
        self.action_dim = config.action_dim

        self.learning_rule = learning_rule
        self.batch_dim = learning_rule.batch_dim
        self.layers = []
        current_seed = config.seed
        for i in range(1, len(config.network_shape)):
            input_dim = config.network_shape[i - 1]
            output_dim = config.network_shape[i]
            if i == 1:
                previous_layer = None
            else:
                previous_layer = self.layers[-1]
            action_flags = torch.zeros(output_dim, device=config.device)
            if (i - 1) == self.action_location:
                action_flags[:self.action_dim] = 1.0
            self.layers.append(ActorNetworkLayer(
                input_dim,
                output_dim,
                learning_rule,
                previous_layer,
                current_seed,
                action_flags, #flagging the neurons that are responsible for the actions
                config
            ))
            if config.seed is not None:
                current_seed += 1
        self.last_input = None
        #self.reward_network_state = self.learning_rule.reward_network.get_zero_state(self.output_dim)
        self.device = config.device
        self.signal_dim = config.signal_dim

    def forward(self, x):
        self.last_input = x
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            #out has dim (batch_dim, output_dim)
            if i == self.action_location:
                actual_out = out[:, :2].clone()
        return actual_out

    def train(self, reward):
        learning_signal = torch.zeros(self.batch_dim, self.output_dim, self.signal_dim, device=self.device)
        for i in range(len(self.layers) - 1, 0, -1):
            learning_signal = self.layers[i].backward(learning_signal, reward)
        # last layer gets passed the input and we ignore the learning signal
        self.layers[0].backward(learning_signal, reward, self.last_input)

    def reset(self):
        for layer in self.layers:
            layer.reset()
        #self.reward_network_state = self.learning_rule.reward_network.get_zero_state(self.output_dim)

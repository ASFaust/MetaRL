from .ActorNetworkLayer import ActorNetworkLayer
import torch

class ActorNetwork:
    def __init__(
            self,
            layer_dims,
            learning_rule,
            weight_limit,
            weight_init,
            sigma_limit,
            sigma_init,
            learning_rate,
            device='cuda',
            seed=None):
        #if seed is none, the parameters are initialized randomly, but the same for every batch
        #if seed is not none, the parameters are initialized randomly, still the same for every batch, but according to the seed
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.learning_rule = learning_rule
        self.batch_dim = learning_rule.batch_dim
        self.layers = []
        current_seed = seed
        for i in range(1,len(layer_dims)):
            input_dim = layer_dims[i-1]
            output_dim = layer_dims[i]
            if i == 1:
                previous_layer = None
            else:
                previous_layer = self.layers[-1]
            self.layers.append(ActorNetworkLayer(
                    input_dim,
                    output_dim,
                    learning_rule,
                    previous_layer,
                    weight_limit,
                    weight_init,
                    sigma_limit,
                    sigma_init,
                    learning_rate,
                    device,
                    current_seed
            ))
            if seed is not None:
                current_seed += 1
        self.out = None
        self.reward_network_state = self.learning_rule.reward_network.get_zero_state(self.output_dim)
        self.device = device

    def forward(self, x):
        self.last_input = x
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train(self, reward):
        #squeeze the reward to (batch_dim,)
        reward = reward.unsqueeze(-1).expand(-1, self.output_dim)[:,:,None]
        node_info = self.layers[-1].get_node_info()
        #has shape (batch_dim, output_dim,  node_info_dim=5)
        reward_network_input = torch.cat((reward, node_info), dim=-1)
        learning_signal, self.reward_network_state = self.learning_rule.reward_network.forward(reward_network_input, self.reward_network_state)
        #learning_signal now has shape (batch_dim, output_dim, signal_dim) (signal dim is 1 most of the time)
        #we now iterate over the layers backwards
        #the first layer gets the learning signal as input
        #the other layers get the learning signal and the output of the previous layer as input
        #the last layer gets the output of the previous layer as input
        #the first layer gets the output of the previous layer as input
        for i in range(len(self.layers) - 1,0,-1):
            layer = self.layers[i]
            learning_signal = layer.backward(learning_signal)
        #last layer gets passed the input and we ignore the learning signal
        self.layers[0].backward(learning_signal, self.last_input)

    def reset(self):
        for layer in self.layers:
            layer.reset() #necessary because the learning rule is stateful
        self.reward_network_state = self.learning_rule.reward_network.get_zero_state(self.output_dim)

    def get_params(self):
        #sigmas,biases and weights from all layers
        params = []
        for layer in self.layers:
            params.append((layer.sigmas, layer.biases, layer.weights))
        return params
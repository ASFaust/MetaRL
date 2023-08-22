from ActorNetworkLayer import ActorNetworkLayer

"""
Actorrnetworklayer:
self,
            input_dim,
            output_dim,
            lrn,
            previous_layer,
            weight_limit=1,
            sigma_limit=1,
            sigma_init=0.1,
            learning_rate=0.1,
            device='cuda'
"""

class ActorNetwork:
    def __init__(
            self,
            layer_dims,
            lrn, #learning rule networks.
            weight_limit=1,
            sigma_limit=1,
            sigma_init=0.1,
            learning_rate=0.1,
            device='cuda',
            seed=None):
        #if seed is none, the parameters are initialized randomly, but the same for every batch
        #if seed is not none, the parameters are initialized randomly, still the same for every batch, but according to the seed
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.lrn = lrn
        self.batch_dim = lrn.batch_dim
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
                    lrn,
                    previous_layer,
                    weight_limit,
                    sigma_limit,
                    sigma_init,
                    learning_rate,
                    device,
                    current_seed
            )
            if seed is not None:
                current_seed += 1

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train(self, reward):
        #we first invoke the lrn.reward_network on the reward and the output of the last layer.
        #y has shape (batch_dim, 1) and contains the reward
        #we need to translate it to a signal for the output layer neurons
        pass
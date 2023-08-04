from ActorNetworkLayer import ActorNetworkLayer

class ActorNetwork:
    def __init__(self, input_dim, output_dim, lrn):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lrn = lrn
        self.batch_dim = lrn.batch_dim
        self.layers = [
            ActorNetworkInputLayer(input_dim, lrn)
        ]
        self.layers.append(
            ActorNetworkLayer(input_dim, output_dim, lrn, self.layers[-1])
        )
        self.layers.append(
            ActorNetworkLayer(output_dim, output_dim, lrn, self.layers[-1])
        )
        self.layers.append(
            ActorNetworkLayer(output_dim, output_dim, lrn, self.layers[-1])
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train(self, y):
        #y has shape (batch_dim, 1) and contains the reward
        #we need to translate it to a signal for the output layer neurons
        pass
# MetaRL - Learning to learn with Genetic Algorithms

A neural network is used as an actor in a reinforcement learning environment. 
Its weights update after every environment step using a learning rule.
The learning rule is parametrized by a small neural network.
The parameters of the learning rule network are optimized by a genetic algorithm to maximize the cumulative reward of the actor. 
An individual of the genetic algorithm is a vector of parameters of the learning rule.
It gets evaluated by running the actor in an environment for a fixed number of steps and summing up the rewards to determine the fitness of the individual.

The RL environment is the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment from the [OpenAI Gym](https://gym.openai.com/) library.

The actor network is a small stochastic feedforward neural network with sigmoid activations and varying stochasticity, which is also learned by the learning rule to allow for varying amounts of exploration.
For training, the actor network uses a backward pass that feeds information through the network much like through a graph neural network, starting with the reward signal at the output neurons of the actor network and propagating a signal backwards through the actor network by passing it repeatedly through the learning rule network, which also computes the weight updates for the actor network.

The weights of the actor network are clipped to a maximum absolute value of 1.0 after every weight update.

The networks are implemented in opencl to allow for fast parallel computation on a gpu, so that many individuals of the genetic algorithm can be evaluated in parallel.
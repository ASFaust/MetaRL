# MetaRL - Learning to learn with Genetic Algorithms

A neural network is used as a controller for a double pendulum simulation, which provides a reward after every time step.
Its weights update after every environment step according to a learning rule that is derived from backpropagation but not equal to it.
The learning rule is parametrized as another, smaller message passing graph neural network.
The parameters of this learning rule network are optimized by a genetic algorithm to maximize the cumulative reward of the actor. 
An individual of the genetic algorithm is a vector of parameters of the learning rule.
It gets evaluated by running the actor in an environment for a fixed number of steps and summing up the rewards to determine the fitness of the individual.

The actor network is a small stochastic feedforward neural network with sigmoid activations and varying stochasticity, which is also learned by the learning rule to allow for varying amounts of exploration.
For training, the actor network uses a backward pass that feeds information through the network much like through a graph neural network, starting with the reward signal at the output neurons of the actor network and propagating a signal backwards through the actor network by passing it repeatedly through the learning rule network, which also computes the weight updates for the actor network.

The networks and pendulum are implemented in pytorch to allow for massive GPU acceleration.

10k individuals only take up around 500MB VRAM.

## Requirements
```
pip install -r requirements.txt
```

## Usage
To show the environments:
```
python3 show_environment.py Pendulum
python3 show_environment.py DoublePendulum
```

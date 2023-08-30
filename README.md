# MetaRL - Learning to learn with Genetic Algorithms

This project utilizes a neural network to control an agent in a reinforcement learning environment. The environment provides an observation, takes an action and produces a reward in each step.

## Overview:

### 1. **Main Neural Network (Actor):**
   - Fully connected feed forward neural network.
   - Controls the agent in the environment.
   - Receives rewards post-action.
   - Modifies its weights following each action using a unique learning rule.
   - This rule is inspired by standard backpropagation (see below).
   - The network's forward pass is stochastic, and its weights and stochasticity are updated after each action.

### 2. **Learning Rule Network (Message Passing Graph Neural Network):**
   - A compact network dictating the learning rule for the main actor.
   - Its weights are optimized using a genetic algorithm to enhance the main neural network's performance.
   - Each genetic algorithm individual corresponds to a distinct weight configuration for the Learning Rule Network.
   - The performance metric for each individual is based on a complete training session of the main neural network using that specific Learning Rule Network configuration.

### 3. **Training Processes:**

#### a. **Main Neural Network's Training:**
   - The network is trained by back-propagating information.
   - The method of this backward flow and subsequent weight adjustments are determined by the Learning Rule Network.

#### b. **Learning Rule Network's Training:**
   - Its weights are learned using a genetic algorithm.
   - Each individual of the genetic algorithm (representing a specific weight configuration of the Learning Rule Network) is assessed based on a full training session of the Main Neural Network.
   - In essence, for the Learning Rule Network to train, the Main Neural Network undergoes repeated training sessions with varying configurations.

### 4. **Technical Specifications:**
   - The entire framework is developed using PyTorch, facilitating rapid GPU computations.
   - Initiating the program with 10,000 genetic algorithm individuals will utilize roughly 500MB of GPU memory.

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

To train the actor network:
```
python3 run_config.py configs/base.yaml
```

To show results of training:
```
python3 show_results.py results/best_model.yaml
```

## Results
It doesn't work (yet). The actor network is not able to learn to control the agent in the environment, which means that the learning rule network is not able to learn a good learning rule.

The main problem in the FoodWorld1 environment seems to be a correlation between the outputs. The strongest individuals learn to move on a diagonal line.
With the separation of negative and positive actions in both the environment and the message passing neural network, the new best behaviour seems to be to slowly creep towards the food, but stop before reaching it. When the food is reached and placed in a new location, the agent starts to oscillate on the diagonal line in the vicinity of the food. 
My idea was that the reward networks would learn something like a Q value based on the action, but now that i think about it, they are only able to learn separate Q values for each action dimension without any correlation between them. This is probably the reason why the agent is not able to learn to move towards the food. maybe we need to instantiate further reward networks for the correlation between the actions, but i wouldnt know how the learning signal from that would be integrated into the message passing neural network.
Maybe if we replace the dense forward pass with a recurrent message passing graph neural network, we could learn the correlation between the actions. But then we would need to learn the weights of the message passing neural network for the forward pass with the genetic algorithm, which would be very slow.
I think one thing to try would be to learn the agent's parameters directly with the genetic algorithm, just to see if there are any implementations errors in the genetic algorithm. 
For that to work, we need to generalize the genetic algorithm to have adapters to both the learning rule network and the actor network.
If that works, we could try to learn the message passing neural network with the genetic algorithm.
I think the problem with my approach is giving the reward to the neurons that output the actions.
right now, the actor network has shape 7,8,2, and 2 is the x,y force for the agent. the reward networks of the message passing neural network create a bridge between these 2 last neurons and the recieved reward. i think a network that outputs some not further designated singular value, which gets hooked to the reward, would be better.
the actions need to be calculated from some other state within the network, not from the output neurons. Maybe the actions need to be part of the input, and they get computed by looking at the learning signal and a separate part of the message passing neural network.

## References
- [Meta-Learning with Differentiable Closed-Form Solvers](https://arxiv.org/abs/1805.08136)
- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)
- [Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments](https://arxiv.org/abs/1710.03641)
- [Learning to learn without gradient descent by gradient descent](https://arxiv.org/abs/1611.03824)
- [Learning to learn with conditional class dependencies](https://openreview.net/pdf?id=BJfOXnActQ)
- [Learning to learn with backpropagation of Hebbian plasticity](https://arxiv.org/abs/1609.02228)
- [PAC Reinforcement Learning for Predictive State Representations](https://arxiv.org/abs/2207.05738)
- [Meta-Learning with Latent Embedding Optimization](https://arxiv.org/abs/1807.05960)

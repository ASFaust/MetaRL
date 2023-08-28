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

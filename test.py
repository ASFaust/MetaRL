import torch
from LearningRule import LearningRule
from ActorNetwork import ActorNetwork
from util import count_parameters

population_size = 1000

params = LearningRule.get_random_params(population_size)

print(count_parameters(params))

lr = LearningRule(params)

actor = ActorNetwork(
            [2, 20, 20, 2],
            lr,
            weight_limit=4,
            sigma_limit=1,
            sigma_init=0.1,
            learning_rate=0.01,
            device='cuda',
            seed=None
)

x = torch.rand(population_size,2,device='cuda')
for i in range(10):
    output = actor.forward(x)

for i in range(4000):
    actor.train(torch.rand(population_size,1,device='cuda'))
    print(i)


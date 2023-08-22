import torch
from LearningRule import LearningRule
from ActorNetwork import ActorNetwork
from util import count_parameters

params = LearningRule.get_random_params(1000)

print(count_parameters(params))

lr = LearningRule(params)

actor = ActorNetwork(
            [2, 9, 2],
            lr,
            weight_limit=4,
            sigma_limit=1,
            sigma_init=0.1,
            learning_rate=0.01,
            device='cuda',
            seed=None
)

x = torch.rand(1000,2,device='cuda')
for i in range(10):
    output = actor.forward(x)
    print(output[0])

actor.train(torch.rand(1000,1,device='cuda'))


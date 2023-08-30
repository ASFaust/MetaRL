import numpy as np
import torch
import sys
import cv2
from environments import Environment
from utils import Config

env = Environment(Config({
    "env_name": sys.argv[1],
    "population_size": 1,
    "device": 'cpu',
    "reward_type": 'value',
    "env_step_size": 0.01
}))

#env.force = torch.tensor([0.0])

ms_per_step = 1
if len(sys.argv) > 2:
    ms_per_step = int(sys.argv[2])
    ms_per_step = max(ms_per_step, 1)

for i in range(10000):
    reward,observation = env.step(env.random_action())
    img = env.render(0)
    cv2.imshow('img', img)
    cv2.waitKey(ms_per_step)
    

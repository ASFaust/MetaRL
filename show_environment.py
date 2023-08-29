import numpy as np
import torch
import sys
import cv2
from environments import get_env
from utils import Config

env = get_env(Config({
    "env_name": sys.argv[1],
    "population_size": 1,
    "device": 'cpu',
    "reward_type": 'height'
}))

env.force = torch.tensor([0.0])

for i in range(10000):
    env.step_euler(0.01)
    img = env.render(0,100)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    

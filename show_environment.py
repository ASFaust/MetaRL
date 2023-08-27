import numpy as np
import torch
import sys
import cv2
from environments import get_env
from utils import Config

env = get_env(Config({
    "env_name": sys.argv[1],
    "population_size": 1,
    "device": 'cpu'
}))

for i in range(10000):
    env.step_rk4(0.01)
    img = env.render(0,100)
    cv2.imshow('img', img)
    cv2.waitKey(1)

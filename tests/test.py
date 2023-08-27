import sys
sys.path.append('../')

import torch
import numpy as np
import cv2
from environments import DoublePendulum, Pendulum
from utils import Config
from networks import LearningRule

config = Config("../configs/base.yaml")
print(config.__dict__)
lr = LearningRule.get_random_params(
    1,
    config.signal_dim,
    config.state_dim,
    config.node_info_dim,
    config.init_std,
    "cpu")

lr = LearningRule(lr)

print("parameter count:" + str(lr.count_params()))

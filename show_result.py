import numpy as np
import torch
import yaml
import sys
import torch
import cv2

from utils import Config
from environments import get_env
from networks import get_actor

#load sys.argv[1] as a yaml file
with open(sys.argv[1], 'r') as f:
    all_info = yaml.load(f, Loader=yaml.FullLoader)

config = Config(all_info["config"])
config.population_size = 1
config.device = 'cpu'

#all_info["population"] is a list of dictionaries
for network in all_info["actor"].keys():
    for param in all_info["actor"][network].keys():
        #convert to tensors
        all_info["actor"][network][param] = torch.tensor(all_info["actor"][network][param])

while True:
    actor = get_actor(all_info["actor"], config)
    env = get_env(config)
    for i in range(config.num_eval_steps):
        print("\rstep: {}/{}".format(i, config.num_eval_steps), end="", flush=True)
        observations = env.get_state()
        actions = actor.forward(observations) * config.actor_force
        env.torque = actions.squeeze()
        env.step_rk4(config.env_step_size)
        reward = env.get_reward()
        actor.train(reward)

        img = env.render(0)
        cv2.imshow('img', img)
        cv2.waitKey(1)
    print("\n.")
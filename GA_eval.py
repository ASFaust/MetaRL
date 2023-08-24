import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import sys

from GA_params import *

from LearningRule import LearningRule
from ActorNetwork import ActorNetwork
from DoublePendulum import DoublePendulum

def plot_data_on_image(image, reward_list, cumulative_reward_list, actor_params):
    # Create a blank canvas of size 900x1200x3
    canvas = 255 * np.ones((900, 1200, 3), dtype=np.uint8)

    # Place the current environment image on the top-left part of the canvas
    canvas[:600, :600] = image

    # Extract individual parameters
    sigmas, biases, weights = zip(*actor_params)
    flattened_weights = np.concatenate([w.flatten().cpu() for w in weights])
    flattened_biases = np.concatenate([b.flatten().cpu() for b in biases])
    flattened_sigmas = np.concatenate([s.flatten().cpu() for s in sigmas])

    # Create plots
    fig, ax = plt.subplots(5, 1, figsize=(6, 9), constrained_layout=True)

    # Plot reward
    ax[0].plot(reward_list, color='blue')
    ax[0].set_title("Reward")

    # Plot cumulative reward
    ax[1].plot(cumulative_reward_list, color='green')
    ax[1].set_title("Cumulative Reward")

    # Histogram for weights with fixed x-axis
    ax[2].hist(flattened_weights, color='red', bins=50, range=(-weight_limit, weight_limit))
    ax[2].set_title("Weights")
    ax[2].set_xlim(-4, 4)

    # Histogram for biases with fixed x-axis
    ax[3].hist(flattened_biases, color='cyan', bins=50, range=(-weight_limit, weight_limit))
    ax[3].set_title("Biases")
    ax[3].set_xlim(-4, 4)

    # Histogram for sigmas with fixed x-axis
    ax[4].hist(flattened_sigmas, color='magenta', bins=50, range=(0, sigma_limit))
    ax[4].set_title("Sigmas")
    ax[4].set_xlim(0, 1)

    # Convert figure to image
    fig.canvas.draw()
    subplot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (3,))

    # Place the subplot image on the right side of the canvas
    canvas[:, 600:] = subplot_image

    # Closing the plot to free up memory
    plt.close()

    return canvas

def evaluate(best_params):
    print("evaluating best actor")

    env = DoublePendulum(1)
    actors = create_actor_network(best_params)

    reward_sum = 0
    reward_list = []
    cumulative_reward_list = []

    for i in range(num_steps):
        print("\rstep: {}/{}".format(i, num_steps), end="", flush=True)
        observations = env.get_state()
        actions = actors.forward(observations) * 5
        env.torque = actions.squeeze()
        env.step_rk4(0.1)
        reward = env.get_reward()
        reward_sum += reward.squeeze().cpu().numpy()

        # Record rewards
        reward_list.append(reward.squeeze().cpu().numpy())
        cumulative_reward_list.append(reward_sum)

        image = env.render(0)

        # Plot the data on the image
        actor_params = actors.get_params()
        combined_image = plot_data_on_image(image, reward_list, cumulative_reward_list, actor_params)
        cv2.imshow("evaluation", combined_image)
        cv2.waitKey(1)
        actors.train(reward)

    print("\ndone")

if __name__ == "__main__":
    # Load the best parameters
    best_params = torch.load(sys.argv[1])

    # Evaluate the best parameters
    evaluate(best_params)
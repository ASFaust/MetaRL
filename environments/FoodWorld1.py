import torch
import cv2
import numpy as np

class FoodWorld1:
    def __init__(self, config):
        self.device = config.device
        self.batch_size = config.population_size
        # Food world is an environment in which an actor has a 2d position and a food particle has a 2d position
        # The actor can move in any direction, and the food particle is stationary
        # The actor is rewarded for getting close to the food particle
        # The actor is penalized for moving
        # 7 states: actor x, actor y, actor x velocity, actor y velocity, food x, food y, food consumed
        # 2 actions: actor x velocity, actor y velocity

        self.state = torch.zeros((self.batch_size, 7), device=self.device)
        # set the actor's initial position to be random
        self.state[:, 0:2] = torch.rand((self.batch_size,2), device=self.device) * 2.0 - 1.0
        # set the food particle's position to be random
        self.state[:, 4:6] = torch.rand((self.batch_size,2), device=self.device) * 2.0 - 1.0
        # start the actor on a trajectory towards the food particle
        self.state[:, 2:4] = self.state[:, 4:6] - self.state[:, 0:2]
        self.min_food_distance = 0.1
        # if the actor is within this distance of the food particle, the food particle is eaten and a new one is spawned
        self.action = torch.zeros((self.batch_size, 2), device=self.device)
        self.dt = config.env_step_size
        self.prev_value = self.get_value()

    def dynamics(self, state):
        x, y, dx, dy, fx, fy, fc = state.T.clone()
        dx *= 0.9
        dy *= 0.9
        dx += self.action[:, 0]
        dy += self.action[:, 1]
        #slight force towards the food particle
        dist = torch.sqrt((x - fx) ** 2 + (y - fy) ** 2)
        dx += (fx - x) * 0.1 / dist
        dy += (fy - y) * 0.1 / dist
        x += dx
        y += dy
        next_state = torch.stack([x, y, dx, dy, fx, fy, fc], dim=1)
        delta = next_state - state
        delta[:, 4:7] *= 0.0
        return delta

    def check_food(self, state):
        x, y, _, _, fx, fy, _ = state.T
        #now check if the actor is close enough to the food particle to eat it.
        distance = torch.sqrt((x - fx) ** 2 + (y - fy) ** 2)
        #we can't use if statements in pytorch, so we use a mask instead
        mask = distance < self.min_food_distance
        #if the actor is close enough to the food particle, then the food particle is eaten and a new one is spawned
        fx[mask] = torch.rand((mask.sum(),), device=self.device) * 2.0 - 1.0
        fy[mask] = torch.rand((mask.sum(),), device=self.device) * 2.0 - 1.0
        #the actor is rewarded for getting close to the food particle
        self.state[mask, 6] += 1.0

    def check_bounds(self, state):
        # check if the actor is out of bounds [-2.0, 2.0] in either x or y, and if so, bounce it back in with a 10% loss of momentum
        # also limit the actor's velocity to abs(10.0)

        x, y, dx, dy, fx, fy, _ = state.T
        mask = torch.abs(dx) > 10.0
        dx[mask] = torch.sign(dx[mask]) * 10.0
        mask = torch.abs(dy) > 10.0
        dy[mask] = torch.sign(dy[mask]) * 10.0

        mask = x < -2.0
        x[mask] = -2.0
        dx[mask] *= -0.9
        mask = x > 2.0
        x[mask] = 2.0
        dx[mask] *= -0.9
        mask = y < -2.0
        y[mask] = -2.0
        dy[mask] *= -0.9
        mask = y > 2.0
        y[mask] = 2.0
        dy[mask] *= -0.9

    def step(self):
        derivatives = self.dynamics(self.state)
        self.state += derivatives * self.dt
        self.check_bounds(self.state)
        self.check_food(self.state)
        self.value = self.get_value()
        self.reward = self.value - self.prev_value
        self.prev_value = self.value
        print(self.state)

    def get_state(self):
        return self.state

    def get_value(self):
        x, y, dx, dy, fx, fy, fc = self.state.T
        distance = torch.sqrt((x - fx) ** 2 + (y - fy) ** 2)
        return fc - distance

    def render(self, index, _):
        #render to numpy array of shape 800,800,3 using opencv
        #world coordinates are -2.2 to 2.2 in both x and y
        #image coordinates are 0 to 800 in both x and y
        #the actor should be a bigger red circle
        #the food particle should be a smaller green circle
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        x, y, dx, dy, fx, fy, fc = self.state[index].cpu().numpy()
        x = int((x + 2.2) / 4.4 * 800)
        y = int((y + 2.2) / 4.4 * 800)
        fx = int((fx + 2.2) / 4.4 * 800)
        fy = int((fy + 2.2) / 4.4 * 800)
        #draw the bounding box around [-2, 2] in both x and y
        bb = int(4.0 / 4.4 * 800)
        d = int(0.2 / 4.4 * 800)
        cv2.rectangle(image, (d, d), (bb + d, bb + d), (255, 255, 255), 1)
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
        cv2.circle(image, (fx, fy), 5, (0, 255, 0), -1)
        #render food consumed
        cv2.putText(image, str(int(fc)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image


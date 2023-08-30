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
        self.min_food_distance = 0.1
        # if the actor is within this distance of the food particle, the food particle is eaten and a new one is spawned
        # action dim is 2, because the actor can move in x and y
        self.action_dim = 4
        self.action_space = torch.tensor([[-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0]], device=self.device) * 0.01
        self.observation_dim = 7
        self.reset()

    def reset(self):
        self.state = torch.zeros((self.batch_size, 7), device=self.device)
        #set the position of the food particle to 1,0.5
        self.state[:, 4] = 1.0
        self.state[:, 5] = 0.5
        #normalize the food particle position to be distance 1 from the center
        distance = torch.norm(self.state[:, 4:6], dim=1, keepdim=True)
        self.state[:, 4:6] = self.state[:, 4:6] / distance
        return self.get_observation()

    def dynamics(self,action):
        # 1. Decomposition
        pos = self.state[:, 0:2]
        vel = self.state[:, 2:4]
        food_pos = self.state[:, 4:6]
        food_count = self.state[:, 6].unsqueeze(1)  # Unsqueeze for matching dimensions

        # 2. Update velocities
        vel *= 0.9
        #action [:,0] is the negative x velocity, action[:,1] is the positive y velocity
        vel[:, 0] += action[:, 0]
        vel[:, 0] += action[:, 1]
        vel[:, 1] += action[:, 2]
        vel[:, 1] += action[:, 3]

        # 3. Cap velocities at a maximum speed
        speed = torch.norm(vel, dim=1, keepdim=True)
        mask = speed > 10.0
        mask_expanded = mask.expand_as(vel)
        vel[mask_expanded] = (vel[mask_expanded] / speed[mask]) * 10.0

        # 4. Update positions
        pos += vel

        # 5. Boundary conditions
        for coord, delta in zip(pos.T, vel.T):
            mask = coord < -2.0
            coord[mask] = -2.0
            delta[mask] *= -0.9

            mask = coord > 2.0
            coord[mask] = 2.0
            delta[mask] *= -0.9

        # Put everything back into self.state
        self.state = torch.cat([pos, vel, food_pos, food_count], dim=1)

    def check_food(self, state):
        x, y, _, _, fx, fy, _ = state.T
        #now check if the actor is close enough to the food particle to eat it.
        distance = torch.sqrt((x - fx) ** 2 + (y - fy) ** 2)
        #we can't use if statements in pytorch, so we use a mask instead
        mask = distance < self.min_food_distance
        #if the actor is close enough to the food particle, then the food particle is eaten and a new one is spawned
        #fx[mask] = torch.rand((mask.sum(),), device=self.device) * 2.0 - 1.0
        #fy[mask] = torch.rand((mask.sum(),), device=self.device) * 2.0 - 1.0
        #deterministically alter the food particle position: rotate it around the origin by 71 degrees
        fx[mask] = fx[mask] * 0.3 - fy[mask] * 0.95
        fy[mask] = fx[mask] * 0.95 + fy[mask] * 0.3
        #and normalize its distance from the origin to be 1
        fx[mask] /= torch.sqrt(fx[mask] ** 2 + fy[mask] ** 2)
        fy[mask] /= torch.sqrt(fx[mask] ** 2 + fy[mask] ** 2)
        #the actor is rewarded for getting close to the food particle
        self.state[mask, 6] += 1.0

    def step(self, action):
        self.dynamics(action)
        self.check_food(self.state)

    def get_observation(self):
        return self.state

    def get_value(self):
        x, y, dx, dy, fx, fy, fc = self.state.T
        distance = torch.sqrt((x - fx) ** 2 + (y - fy) ** 2)
        return fc - distance

    def render(self, index):
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


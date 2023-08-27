import torch
import cv2
import numpy as np

class DoublePendulum:
    @staticmethod
    def get_params(batch_size,
                   l1_min=1.0,
                   l1_max=1.0,
                   l2_min=1.0,
                   l2_max=1.0,
                   m1_min=1.0,
                   m1_max=1.0,
                   m2_min=1.0,
                   m2_max=1.0,
                   g_min=9.81,
                   g_max=9.81,
                   damping_min=0.1,
                   damping_max=0.1,
                   device='cuda'):
        l1 = torch.rand((batch_size,), device=device) * (l1_max - l1_min) + l1_min
        l2 = torch.rand((batch_size,), device=device) * (l2_max - l2_min) + l2_min
        m1 = torch.rand((batch_size,), device=device) * (m1_max - m1_min) + m1_min
        m2 = torch.rand((batch_size,), device=device) * (m2_max - m2_min) + m2_min
        g = torch.rand((batch_size,), device=device) * (g_max - g_min) + g_min
        damping = torch.rand((batch_size,), device=device) * (damping_max - damping_min) + damping_min
        return l1, l2, m1, m2, g, damping

    def __init__(self, batch_size, params=None, device='cuda'):
        if params is None:
            params = DoublePendulum.get_params(batch_size, device=device)
        self.l1, self.l2, self.m1, self.m2, self.g, self.damping = params
        self.torque = torch.zeros((batch_size, ), device = device) #
        self.device = device
        self.batch_size = batch_size
        assert self.batch_size == self.l1.shape[0] == self.l2.shape[0] == self.m1.shape[0] == self.m2.shape[0] == self.g.shape[0], "batch_size must be the same for all parameters"
        self.state = torch.zeros((self.batch_size, 4), device=device)  # [theta1, omega1, theta2, omega2]
        self.state[:,0] = torch.zeros((self.batch_size, ), device=device) * np.pi * 1.00001
        self.state[:,2] = torch.zeros((self.batch_size, ), device=device) * np.pi
        self.waitkey = -1
        theta1, omega1, theta2, omega2 = self.state.T
        self.last_value = 0

    def dynamics(self, state):
        theta1, omega1, theta2, omega2 = state.T
        delta_theta = theta1 - theta2
        den1 = (2.0 * self.m1 + self.m2) - self.m2 * torch.cos(2.0 * delta_theta)
        den2 = den1  # If you intended different denominators, modify this
        #broken in some kind of way.

        a1 = (-self.g * (2 * self.m1 + self.m2) * torch.sin(theta1) - self.m2 * self.g * torch.sin(
            theta1 - 2 * theta2) - 2 * torch.sin(delta_theta) * self.m2 * (
                      omega2 ** 2 * self.l2 + omega1 ** 2 * self.l1 * torch.cos(
                  delta_theta)) + self.torque) / self.l1 / den1 - self.damping * omega1
        a2 = (2 * torch.sin(delta_theta) * (
                omega1 ** 2 * self.l1 * (self.m1 + self.m2) + self.g * (self.m1 + self.m2) * torch.cos(
            theta1) + omega2 ** 2 * self.l2 * self.m2 * torch.cos(delta_theta))) / self.l2 / den2 - self.damping * omega2

        return torch.stack([omega1, a1, omega2, a2], dim=1)

    def step_euler(self, dt = 0.01):
        # Simple integration using Euler's method
        derivatives = self.dynamics(self.state)
        self.state += dt * derivatives
        self.wrap_angles()
        self.clamp_speed()

    def step_rk4(self, dt = 0.01):
        # Integration using Runge-Kutta 4
        k1 = dt * self.dynamics(self.state)
        k2 = dt * self.dynamics(self.state + 0.5 * k1)
        k3 = dt * self.dynamics(self.state + 0.5 * k2)
        k4 = dt * self.dynamics(self.state + k3)

        self.state += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        self.wrap_angles()
        self.clamp_speed()

    def wrap_angles(self):
        # Wrap angles between 0 and 2pi
        self.state[:, 0] = torch.fmod(self.state[:, 0], 2.0 * np.pi)
        self.state[:, 2] = torch.fmod(self.state[:, 2], 2.0 * np.pi)

    def clamp_speed(self, max_speed = 10.0):
        # Clamp angular velocities
        self.state[:, 1] = torch.clamp(self.state[:, 1], -max_speed, max_speed)
        self.state[:, 3] = torch.clamp(self.state[:, 3], -max_speed, max_speed)

    def get_state(self):
        #get sin and cos of theta1 and theta2 to have smooth transition from 2pi to 0
        theta1, omega1, theta2, omega2 = self.state.T
        sin_theta1 = torch.sin(theta1)
        cos_theta1 = torch.cos(theta1)
        sin_theta2 = torch.sin(theta2)
        cos_theta2 = torch.cos(theta2)
        return torch.stack([sin_theta1, cos_theta1, omega1, sin_theta2, cos_theta2, omega2], dim=1)
        #this returns a tensor of shape (batch_size, 6) with the following columns:
        #sin(theta1), cos(theta1), omega1, sin(theta2), cos(theta2), omega2

    def get_reward(self):
        #reward is the sum of the cosines of the angles minus the absolute value of the angular velocities
        theta1, omega1, theta2, omega2 = self.state.T
        value = (1.0 - torch.cos(theta1)) + (1.0 - torch.cos(theta2)) #- torch.abs(omega1) - torch.abs(omega2)
        reward = value - self.last_value
        #make negative rewards
        reward[reward < 0] *= 1.1
        self.last_value = value
        return reward

    def render(self,index, scale=100):
        theta1, omega1, theta2, omega2 = self.state[index]

        # Calculate positions
        x0, y0 = 300, 300  # Center of the window
        x1 = x0 + scale * self.l1[index] * torch.sin(theta1).item()
        y1 = y0 + scale * self.l1[index] * torch.cos(theta1).item()
        x2 = x1 + scale * self.l2[index] * torch.sin(theta2).item()
        y2 = y1 + scale * self.l2[index] * torch.cos(theta2).item()

        # Create an empty canvas
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)

        # Draw pendulum
        cv2.circle(canvas, (int(x0), int(y0)), 5, (0, 255, 0), -1)  # Origin
        cv2.circle(canvas, (int(x1), int(y1)), 5, (0, 0, 255), -1)  # Joint 1
        cv2.circle(canvas, (int(x2), int(y2)), 5, (255, 0, 0), -1)  # Joint 2

        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)  # Rod 1
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)  # Rod 2

        return canvas
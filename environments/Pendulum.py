import torch
import cv2
import numpy as np

class Pendulum:
    @staticmethod
    def get_params(batch_size,
                   l_min=1.0,
                   l_max=1.0,
                   m_min=1.0,
                   m_max=1.0,
                   g_min=9.81,
                   g_max=9.81,
                   damping_min=0.1,
                   damping_max=0.1,
                   device='cuda'):
        l = torch.rand((batch_size,), device=device) * (l_max - l_min) + l_min
        m = torch.rand((batch_size,), device=device) * (m_max - m_min) + m_min
        g = torch.rand((batch_size,), device=device) * (g_max - g_min) + g_min
        damping = torch.rand((batch_size,), device=device) * (damping_max - damping_min) + damping_min
        return l, m, g, damping

    def __init__(self, batch_size, params=None, device='cuda'):
        if params is None:
            params = Pendulum.get_params(batch_size, device=device)
        self.l, self.m, self.g, self.damping = params
        self.torque = torch.zeros((batch_size,), device=device)
        self.device = device
        self.batch_size = batch_size
        assert self.batch_size == self.l.shape[0] == self.m.shape[0] == self.g.shape[0], "batch_size must be the same for all parameters"
        self.state = torch.ones((self.batch_size, 2), device=device)  # [theta, omega]
        self.state[:, 0] = torch.ones((self.batch_size,), device=device) * np.pi * 1.00001
        self.last_value = None

    def dynamics(self, state):
        theta, omega = state.T
        a = (-self.g / self.l) * torch.sin(theta) + (self.torque / (self.m * self.l ** 2)) - self.damping * omega
        return torch.stack([omega, a], dim=1)

    def step_euler(self, dt=0.01):
        derivatives = self.dynamics(self.state)
        self.state += dt * derivatives
        self.wrap_angles()
        self.clamp_speed()

    def step_rk4(self, dt=0.01):
        k1 = dt * self.dynamics(self.state)
        k2 = dt * self.dynamics(self.state + 0.5 * k1)
        k3 = dt * self.dynamics(self.state + 0.5 * k2)
        k4 = dt * self.dynamics(self.state + k3)

        self.state += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        self.wrap_angles()
        self.clamp_speed()

    def wrap_angles(self):
        self.state[:, 0] = (self.state[:, 0] + np.pi) % (2.0 * np.pi) - np.pi

    def clamp_speed(self, max_speed=10.0):
        self.state[:, 1] = torch.clamp(self.state[:, 1], -max_speed, max_speed)

    def get_state(self):
        theta, omega = self.state.T
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        return torch.stack([sin_theta, cos_theta, omega], dim=1)

    def get_reward(self):
        theta, omega = self.state.T
        value = (1.0 - torch.cos(theta))
        if self.last_value is None:
            self.last_value = value
            return torch.zeros((self.batch_size,), device=self.device)
        reward = value - self.last_value
        reward[reward < 0] *= 1.1
        self.last_value = value
        return reward

    def render(self, index, scale=100):
        theta, omega = self.state[index]
        x0, y0 = 300, 300
        x1 = x0 + scale * self.l[index] * torch.sin(theta).item()
        y1 = y0 + scale * self.l[index] * torch.cos(theta).item()
        canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.circle(canvas, (int(x0), int(y0)), 5, (0, 255, 0), -1)  # Origin
        cv2.circle(canvas, (int(x1), int(y1)), 5, (0, 0, 255), -1)  # Joint
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)  # Rod
        return canvas

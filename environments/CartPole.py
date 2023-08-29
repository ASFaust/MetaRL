import torch
import cv2
import numpy as np


class CartPole:
    @staticmethod
    def get_params(batch_size,
                   l_min=1.0,
                   l_max=1.0,
                   m_cart_min=1.0,
                   m_cart_max=1.0,
                   m_pole_min=0.5,
                   m_pole_max=0.5,
                   mu_cart_min=0.1,
                   mu_cart_max=0.1,
                   mu_pole_min=0.1,
                   mu_pole_max=0.1,
                   g_min=-9.81,
                   g_max=-9.81,
                   device='cuda'):
        l = torch.rand((batch_size,), device=device) * (l_max - l_min) + l_min
        m_cart = torch.rand((batch_size,), device=device) * (m_cart_max - m_cart_min) + m_cart_min
        m_pole = torch.rand((batch_size,), device=device) * (m_pole_max - m_pole_min) + m_pole_min
        g = torch.rand((batch_size,), device=device) * (g_max - g_min) + g_min
        mu_cart = torch.rand((batch_size,), device=device) * (mu_cart_max - mu_cart_min) + mu_cart_min
        mu_pole = torch.rand((batch_size,), device=device) * (mu_pole_max - mu_pole_min) + mu_pole_min

        return l, m_cart, m_pole, g, mu_cart, mu_pole

    def __init__(self, batch_size, params=None, reward_type="height", device='cuda'):
        if params is None:
            params = CartPole.get_params(batch_size, device=device)

        self.l, self.m_cart, self.m_pole, self.g, self.mu_cart, self.mu_pole = params
        self.force = torch.zeros((batch_size,), device=device)
        self.device = device
        self.batch_size = batch_size
        assert self.batch_size == self.l.shape[0] == self.m_cart.shape[0] == self.m_pole.shape[0] == self.g.shape[
            0], "batch_size must be the same for all parameters"

        # [cart_position, cart_velocity, pole_angle, pole_velocity]
        self.state = torch.zeros((self.batch_size, 4), device=device)
        self.state[:, 2] = 0.0001

    def dynamics(self, state):
        x, x_dot, theta, theta_dot = state.T

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        F = self.force
        g = self.g
        l = self.l
        m_c = self.m_cart
        m_p = self.m_pole
        mu_c = self.mu_cart  # Coefficient of friction for the cart
        mu_p = self.mu_pole  # Coefficient of friction for the pole's pivot

        denominator = m_c + m_p * sin_theta ** 2

        x_dotdot = (F + m_p * sin_theta * (l * theta_dot ** 2 - g * cos_theta) - mu_c * x_dot) / denominator

        theta_dotdot = (-F * cos_theta + m_p * l * theta_dot ** 2 * cos_theta * sin_theta - (
                    m_c + m_p) * g * sin_theta - mu_p * theta_dot) / (l * denominator)

        #add some force when position is too far from center (-1,1)
        if x < -1.0 or x > 1.0:
            x_dotdot += x * 0.1

        return torch.stack([x_dot, -x_dotdot, theta_dot, theta_dotdot], dim=1)

    def step_euler(self, dt=0.01):
        derivatives = self.dynamics(self.state)
        self.state += dt * derivatives

    def wrap_angles(self):
        self.state[:, 2] = (self.state[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

    def get_state(self):
        x, x_dot, theta, theta_dot = self.state.T
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        return torch.stack([x, x_dot, sin_theta, cos_theta, theta_dot], dim=1)

    def reward_height(self):
        theta, _ = self.state[:, 2], self.state[:, 3]
        return 1.0 - torch.cos(theta)

    def render(self, index, scale=100):
        x, _, theta, _ = self.state[index]
        x0, y0 = 300 + scale * x.item(), 400
        x1 = x0 + scale * self.l[index] * torch.sin(theta).item()
        y1 = y0 - scale * self.l[index] * torch.cos(theta).item()
        canvas = np.zeros((800, 600, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (int(x0 - 20), int(y0)), (int(x0 + 20), int(y0 + 20)), (0, 255, 0), -1)  # Cart
        cv2.circle(canvas, (int(x1), int(y1)), 5, (0, 0, 255), -1)  # Pole tip
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)  # Pole
        return canvas

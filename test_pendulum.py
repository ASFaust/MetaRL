from DoublePendulum import DoublePendulum
import cv2
import numpy as np

sum_reward = np.zeros((2,))
pendulum = DoublePendulum(2)
for i in range(1000):
    pendulum.step_rk4(0.1)
    image = pendulum.render(0)
    cv2.imshow('image',image)
    cv2.waitKey(100)
    reward = pendulum.get_reward().cpu().numpy()
    sum_reward += reward
    print(reward)
    print(sum_reward)
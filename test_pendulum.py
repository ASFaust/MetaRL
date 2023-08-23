from DoublePendulum import DoublePendulum

pendulum = DoublePendulum(2)
for i in range(1000):
    pendulum.step_rk4(0.1)
    pendulum.render(0)
    print(pendulum.get_reward())
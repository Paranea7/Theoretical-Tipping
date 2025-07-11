import numpy as np
import matplotlib.pyplot as plt

# 定义参数
k = 10
alpha = 1
beta = 1
dt = 0.01  # 步长
time_end = 100  # 仿真结束时间
num_steps = int(time_end / dt)

# RK4 方法
def rk4(x0, r_value, dt, num_steps):
    x = x0
    for i in range(num_steps):
        k1 = dt * (r_value * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
        k2 = dt * (r_value * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                   (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
        k3 = dt * (r_value * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                   (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
        k4 = dt * (r_value * (x + k3) * (1 - (x + k3) / k) -
                   (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x  # 返回最后的 x 值作为稳态数值

# 生成 r 值，范围从 0 到 1，步长为 0.02
r_values_increase = np.linspace(0, 1, int(1 / 0.01) + 1)
r_values_decrease = np.linspace(1, 0, int(1 / 0.01) + 1)

x_stable_rk4_increase = []
x_stable_rk4_decrease = []

# 计算增加 r 值的稳态 x
for r_value in r_values_increase:
    steady_x_rk4 = rk4(x0=0.1, r_value=r_value, dt=0.01, num_steps=num_steps)
    x_stable_rk4_increase.append(steady_x_rk4)

# 计算减少 r 值的稳态 x
for r_value in r_values_decrease:
    steady_x_rk4 = rk4(x0=9, r_value=r_value, dt=0.01, num_steps=num_steps)
    x_stable_rk4_decrease.append(steady_x_rk4)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(r_values_increase, x_stable_rk4_increase, marker='o', label='RK4 Steady state x (Increasing r)', linestyle='-')
plt.plot(r_values_decrease, x_stable_rk4_decrease, marker='x', label='RK4 Steady state x (Decreasing r)', linestyle='--')
plt.xlabel('r')
plt.ylabel('Steady state x')
plt.title('Dynamics of Steady State x vs r')
plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度
plt.yticks(np.arange(0, 11, 1))  # 根据需要调整 y 轴刻度
plt.legend()
plt.grid()
plt.show()
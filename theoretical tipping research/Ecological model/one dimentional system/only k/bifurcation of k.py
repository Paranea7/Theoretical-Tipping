import numpy as np
import matplotlib.pyplot as plt

# 定义参数
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

# 生成 k 值，范围从 1 到 20
k_values = np.linspace(0.1, 15, 200)  # k 的值范围
r_fixed = 0.47 # 固定的 r 值

x_stable_rk4 = []

# 计算稳态 x 对于不同的 k 值
for k in k_values:
    steady_x_rk4 = rk4(x0=5, r_value=r_fixed, dt=dt, num_steps=num_steps)
    x_stable_rk4.append(steady_x_rk4)

# 绘制 k-x 平面的结果
plt.figure(figsize=(12, 6))
plt.plot(k_values, x_stable_rk4, marker='o', label=f'r = {r_fixed}')

plt.xlabel('k')
plt.ylabel('Steady state x')
plt.title('Dynamics of Steady State x vs k (fixed r = 0.47)')
plt.xticks(np.arange(1, 16, 1))  # 设置 k 轴刻度
plt.yticks(np.arange(0, 14))  # 根据需要调整 y 轴刻度
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
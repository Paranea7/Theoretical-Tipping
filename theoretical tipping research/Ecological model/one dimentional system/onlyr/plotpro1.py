import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 定义参数
k = 10
alpha = 1
beta = 1
dt = 0.01  # 步长
time_end = 100  # 仿真结束时间
num_steps = int(time_end / dt)

# 向量化的 RK4 方法
@njit
def rk4(x0, amplitude, frequency, dt, num_steps):
    def r_func(t):
        return 0.47 + amplitude * np.sin(frequency * t)

    x = x0
    x_values = np.zeros(num_steps)  # 存储每个时间步的 x 值
    for i in range(num_steps):
        r_value = r_func(i * dt)  # 计算当前时间的 r(t)
        k1 = dt * (r_value * x * (1 - x / k) - (beta * x ** 2) / (alpha ** 2 + x ** 2))
        k2 = dt * (r_value * (x + 0.5 * k1) * (1 - (x + 0.5 * k1) / k) -
                   (beta * (x + 0.5 * k1) ** 2) / (alpha ** 2 + (x + 0.5 * k1) ** 2))
        k3 = dt * (r_value * (x + 0.5 * k2) * (1 - (x + 0.5 * k2) / k) -
                   (beta * (x + 0.5 * k2) ** 2) / (alpha ** 2 + (x + 0.5 * k2) ** 2))
        k4 = dt * (r_value * (x + k3) * (1 - (x + k3) / k) -
                   (beta * (x + k3) ** 2) / (alpha ** 2 + (x + k3) ** 2))
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_values[i] = x  # 记录每个时间步的 x 值
    return x_values  # 返回所有时间步的 x 值

# 设置频率和振幅的组合
frequencies = [0.02, 0.05, 0.1]
amplitudes = [0.05, 0.1, 0.2]

# 生成初值
x0_values = np.linspace(0, 9, 910)  # 生成初始值
time_range = np.arange(0, time_end, dt)  # 时间范围

# 创建图形和子图
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Dynamics of x over time for varying r(t)', fontsize=16)

# 计算每个组合的演化并绘图
for i, frequency in enumerate(frequencies):
    for j, amplitude in enumerate(amplitudes):
        for x0 in x0_values:
            all_x_values = rk4(x0=x0, amplitude=amplitude, frequency=frequency, dt=dt, num_steps=num_steps)
            color = 'red' if x0 > 2.36 else 'blue'
            axs[j, i].plot(time_range, all_x_values, color=color, alpha=0.5)

        # 添加标签和标题
        axs[j, i].set_title(f'Freq: {frequency}, Amp: {amplitude}')
        axs[j, i].set_xlabel('Time')
        axs[j, i].set_ylabel('x')
        axs[j, i].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.32')
        axs[j, i].grid()


plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整图形以适应标签
plt.show()
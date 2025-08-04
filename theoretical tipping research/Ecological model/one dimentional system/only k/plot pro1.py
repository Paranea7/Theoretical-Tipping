import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 定义参数
k0 = 10  # 基础 k 值
alpha = 1
beta = 1
dt = 0.01  # 时间步长
time_end = 100  # 仿真结束时间
num_steps = int(time_end / dt)

# 向量化的 RK4 方法
@njit
def rk4(x0, amplitude_type, frequency, dt, num_steps):
    x = x0
    x_values = np.zeros(num_steps)  # 存储每个时间步的 x 值
    for i in range(num_steps):
        t = i * dt

        # 根据选择的振幅类型计算振幅
        if amplitude_type == 0:  # 1/t
            amplitude = 1 / t if t != 0 else 0
        elif amplitude_type == 1:  # 1/exp(t)
            amplitude = 1 / np.exp(t)
        elif amplitude_type == 2:  # 固定为 2
            amplitude = 2

        # 计算动态 k(t)
        k = k0 + amplitude * np.sin(frequency * t)  # 动态 k 按振幅和频率变化
        r_value = 0.47  # 固定 r 值

        # RK4 计算步骤
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

# 设置频率和振幅
frequencies = [0.02, 0.05, 0.1]  # 调整为三个频率
amplitude_types = [0, 1, 2]  # 1/t, 1/exp(t), 固定为 2

# 创建图形和子图
fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # 4行3列
fig.suptitle('Dynamics of x over time for varying k(t)', fontsize=16)

# 开始绘制图形
for i, frequency in enumerate(frequencies):
    for j, amplitude_type in enumerate(amplitude_types):
        for x0 in np.linspace(0, 9, 910):  # 在 0 到 9 之间取 910 个初始值
            all_x_values = rk4(x0=x0, amplitude_type=amplitude_type, frequency=frequency, dt=dt, num_steps=num_steps)
            color = 'red' if x0 > 2.36 else 'blue'
            # 绘制到相应的子图
            axs[j, i].plot(np.arange(0, time_end, dt), all_x_values, color=color, alpha=0.5)

        # 添加标签和标题
        axs[j, i].set_title(f'Freq: {frequency}, Amp: {["1/t", "1/exp(t)", "2"][amplitude_type]}')
        axs[j, i].set_xlabel('Time')
        axs[j, i].set_ylabel('x')
        axs[j, i].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.36')
        axs[j, i].grid()

# 把最后一排的频率设置为固定 0.05
for j, amplitude_type in enumerate([0, 1]):
    frequency = 0.05  # 固定频率
    for x0 in np.linspace(0, 9, 910):
        all_x_values = rk4(x0=x0, amplitude_type=2, frequency=frequency, dt=dt, num_steps=num_steps)
        color = 'red' if x0 > 2.36 else 'blue'
        axs[3, j].plot(np.arange(0, time_end, dt), all_x_values, color=color, alpha=0.5)

    # 添加标签和标题
    axs[3, j].set_title(f'Amp: 2, Freq: {["1/t", "1/exp(t)"][j]}')
    axs[3, j].set_xlabel('Time')
    axs[3, j].set_ylabel('x')
    axs[3, j].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.36')
    axs[3, j].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整图形以适应标签
plt.show()
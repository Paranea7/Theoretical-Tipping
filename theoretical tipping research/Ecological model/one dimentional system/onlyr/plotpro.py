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
def rk4(x0, amplitude_type, frequency, dt, num_steps):
    x = x0
    x_values = np.zeros(num_steps)  # 存储每个时间步的 x 值
    for i in range(num_steps):
        t = i * dt
        # 根据选择的振幅类型计算 r(t)
        if amplitude_type == 0:  # 1/t
            amplitude = 1 / t if t != 0 else 0
        elif amplitude_type == 1:  # 1/exp(t)
            amplitude = 1 / np.exp(t)

        r_value = 0.47 + amplitude * np.sin(frequency * t)  # 计算 r(t)

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

# 向量化的 RK4 方法
@njit
def rk41(x0, amplitude, frequency, dt, num_steps):
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
# 设置频率
frequencies = [0.02, 0.05, 0.1]

# 创建图形和子图
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Dynamics of x over time for varying r(t)', fontsize=16)

# 第一组图：振幅为 1/t 和 1/exp(t)
for i, frequency in enumerate(frequencies):
    for j, amplitude_type in enumerate([0, 1]):  # 0 为 1/t，1 为 1/exp(t)
        for x0 in np.linspace(0, 9, 910):
            all_x_values = rk4(x0=x0, amplitude_type=amplitude_type, frequency=frequency, dt=dt, num_steps=num_steps)
            color = 'red' if x0 > 2.36 else 'blue'
            axs[j, i].plot(np.arange(0, time_end, dt), all_x_values, color=color, alpha=0.5)

        # 添加标签和标题
        axs[j, i].set_title(f'Freq: {frequency}, Amp: {["1/t", "1/exp(t)"][amplitude_type]}')
        axs[j, i].set_xlabel('Time')
        axs[j, i].set_ylabel('x')
        axs[j, i].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.36')
        axs[j, i].grid()


# 描绘第二组图：振幅为 0.05，频率为 1/t 和 1/exp(t)
for i, amplitude_type in enumerate([0, 1]):
    frequency = 0.05  # 振幅固定为 0.05
    for x0 in np.linspace(0, 9, 910):
        all_x_values = rk4(x0=x0, amplitude_type=amplitude_type, frequency=frequency, dt=dt, num_steps=num_steps)
        color = 'red' if x0 > 2.36 else 'blue'
        axs[2, i].plot(np.arange(0, time_end, dt), all_x_values, color=color, alpha=0.5)

    # 添加标签和标题
    axs[2, i].set_title(f'Amp: 0.05, Freq: {["1/t", "1/exp(t)"][amplitude_type]}')
    axs[2, i].set_xlabel('Time')
    axs[2, i].set_ylabel('x')
    axs[2, i].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.36')
    axs[2, i].grid()

# 第九个子图：振幅为 0.1，频率固定为 0.05
frequency_fixed = 0.1
amplitude_fixed = 0.05

# 绘制振幅为 0.1 的图像
for x0 in np.linspace(0, 9, 910):
    all_x_values = rk41(x0=x0, amplitude=amplitude_fixed, frequency=frequency_fixed, dt=dt, num_steps=num_steps)  # 使用 -1 表示特殊的振幅
    color = 'red' if x0 > 2.36 else 'blue'
    axs[2, 2].plot(np.arange(0, time_end, dt), all_x_values, color=color, alpha=0.5)

# 添加第九个子图的标签和标题
axs[2, 2].set_title(f'Amp: {amplitude_fixed}, Freq: {frequency_fixed}')
axs[2, 2].set_xlabel('Time')
axs[2, 2].set_ylabel('x')
axs[2, 2].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.36')
axs[2, 2].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整图形以适应标签
plt.show()
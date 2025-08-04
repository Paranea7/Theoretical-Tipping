import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 定义参数
alpha = 1
beta = 1
dt = 0.01  # 步长
time_end = 100  # 仿真结束时间
num_steps = int(time_end / dt)

# 向量化的 RK4 方法，合并 k_func 为 rk4 的部分
@njit
def rk4(x0, amplitude, frequency, dt, num_steps):
    x = x0
    x_values = np.zeros(num_steps)  # 存储每个时间步的 x 值
    for i in range(num_steps):
        t = i * dt
        r_value = 0.47  # 固定 r 值
        k = 10 + amplitude * np.sin(frequency * t)  # 直接在这里计算 k(t)

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

# 生成初始值
x0_values = np.linspace(0, 9, 910)  # 生成初始值
frequencies = [0.02, 0.03, 0.05]  # 频率列表
amplitudes = [1, 1.5, 2]  # 振幅列表

# 创建子图
fig, axs = plt.subplots(len(amplitudes), len(frequencies), figsize=(15, 10))
fig.suptitle('Dynamics of x over time with varying k(t) parameters', fontsize=16)

# 计算并绘制每个子图
for i, amplitude in enumerate(amplitudes):
    for j, frequency in enumerate(frequencies):
        all_x_values = np.zeros((len(x0_values), num_steps))  # 存储每个初始条件的演化
        for k in range(len(x0_values)):
            all_x_values[k] = rk4(x0=x0_values[k], amplitude=amplitude, frequency=frequency, dt=dt, num_steps=num_steps)

        for k in range(len(x0_values)):
            # 根据初始值的大小选择颜色
            color = 'red' if x0_values[k] > 2.36 else 'blue'
            axs[i, j].plot(np.arange(0, time_end, dt), all_x_values[k], color=color, alpha=0.5)

        # 添加标签和标题
        axs[i, j].set_title(f'Amp: {amplitude}, Freq: {frequency}')
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('x')
        axs[i, j].axhline(y=2.36, color='grey', linestyle='--', label='x = 2.36')  # 添加参考线
        axs[i, j].grid()
        axs[i, j].legend(["x > 2.36", "x ≤ 2.36"], loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整图形以适应标签
plt.show()
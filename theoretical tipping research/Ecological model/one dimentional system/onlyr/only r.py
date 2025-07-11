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
def rk4(x0, r_func, dt, num_steps):
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

# 固定 r 和生成初始条件
@njit
def r_func(t):
    return 0.47 + np.exp(-t) * np.sin(0.8 * t)

# 生成初始值
x0_values = np.linspace(0, 7, 710)  # 生成初始值
all_x_values = np.zeros((len(x0_values), num_steps))  # 存储每个初始条件的演化

# 计算每个初始值的演化
for j in range(len(x0_values)):
    all_x_values[j] = rk4(x0=x0_values[j], r_func=r_func, dt=dt, num_steps=num_steps)

# 绘制结果
plt.figure(figsize=(12, 6))

for i in range(len(x0_values)):
    # 根据初始值的大小选择颜色
    color = 'red' if x0_values[i] > 2.32 else 'blue'
    plt.plot(np.arange(0, time_end, dt), all_x_values[i], color=color, alpha=0.5)

# 添加标签和标题
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Dynamics of x over time for varying r(t)')
plt.axhline(y=2.32, color='grey', linestyle='--', label='x = 2.32')  # 添加参考线
plt.grid()
plt.legend(["x > 2.32", "x ≤ 2.32"], loc='upper left')
plt.tight_layout()  # 调整图形以适应标签
plt.show()
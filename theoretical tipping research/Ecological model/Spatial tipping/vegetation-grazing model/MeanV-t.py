import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 定义参数
V_c = 20.
D = 0.001
dt = 0.01  # 步长
time_end = 100  # 仿真结束时间
num_steps = int(time_end / dt)
c = 25  # 对于新模型的常数 c

# 向量化的 RK4 方法
@njit
def rk4(V0, r_func, dt, num_steps):
    V = V0
    V_values = np.zeros(num_steps)  # 存储每个时间步的 V 值
    for i in range(num_steps):
        r_value = r_func(i * dt)  # 计算当前时间的 r(t)
        # 根据新的动态方程计算 k1, k2, k3, k4
        k1 = dt * (r_value * V * (1 - V / V_c) - (c * V ** 2) / (V ** 2 + 1) + D * laplacian(V))
        k2 = dt * (r_value * (V + 0.5 * k1) * (1 - (V + 0.5 * k1) / V_c) - (c * (V + 0.5 * k1) ** 2) / ((V + 0.5 * k1) ** 2 + 1) + D * laplacian(V + 0.5 * k1))
        k3 = dt * (r_value * (V + 0.5 * k2) * (1 - (V + 0.5 * k2) / V_c) - (c * (V + 0.5 * k2) ** 2) / ((V + 0.5 * k2) ** 2 + 1) + D * laplacian(V + 0.5 * k2))
        k4 = dt * (r_value * (V + k3) * (1 - (V + k3) / V_c) - (c * (V + k3) ** 2) / ((V + k3) ** 2 + 1) + D * laplacian(V + k3))

        V += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        V_values[i] = V  # 记录每个时间步的 V 值
    return V_values  # 返回所有时间步的 V 值

# 计算拉普拉斯算子
@njit
def laplacian(V):
    return (
        np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
        np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V
    )

# 固定 r 和生成初始条件
@njit
def r_func(t):
    return 4.7  # 固定的 r 值

# 生成初始值
initial_conditions = np.linspace(5, 20, 710)  # 初始值范围
mean_values = np.zeros(num_steps)  # 存储每个时间步的均值

# 计算每个初始值的演化并计算均值
for V0 in initial_conditions:
    V_values = rk4(V0, r_func, dt, num_steps)
    mean_values += V_values  # 累加每个条件的演化

mean_values /= len(initial_conditions)  # 计算均值

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, time_end, dt), mean_values, color='blue', alpha=0.8)

# 添加标签和标题
plt.xlabel('Time')
plt.ylabel('Mean V')
plt.title('Mean V over time for varying initial conditions')
plt.grid()
plt.tight_layout()  # 调整图形以适应标签
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 定义微分方程组
def dynamic_system(y, r, K, d):
    x1, x2 = y
    dx1dt = r * x1 * (1 - x1 / K) - (x1 ** 2) / (1 + x1 ** 2) - d * x2
    dx2dt = r * x2 * (1 - x2 / K) - (x2 ** 2) / (1 + x2 ** 2) - d * x1
    return np.array([dx1dt, dx2dt])

# 定义向量化的RK4方法
def rk4_step(f, y, dt, r, K, d):
    k1 = f(y, r, K, d)
    k2 = f(y + 0.5 * dt * k1, r, K, d)
    k3 = f(y + 0.5 * dt * k2, r, K, d)
    k4 = f(y + dt * k3, r, K, d)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# 参数设置
K = 10.0  # 饱和度
d = 0.1   # 交互作用强度
dt = 0.01 # 时间步长
t_max = 50 # 最大时间
r = 1  # 选择一个具体的 r 值

# 存储结果
time_steps = np.arange(0, t_max, dt)
x1_values = []
x2_values = []

# 重置状态，设定不同的初始条件
y = np.array([1.0, 0.5])  # x1 初值 1.0, x2 初值 0.5
for t in time_steps:
    x1_values.append(y[0])
    x2_values.append(y[1])
    y = rk4_step(dynamic_system, y, dt, r, K, d)
    y = np.maximum(y, 0)  # 确保 x1 和 x2 都是非负的

# 将结果转换为 NumPy 数组（可选）
x1_values = np.array(x1_values)
x2_values = np.array(x2_values)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(time_steps, x1_values, label='x1', color='b', linestyle='-')
plt.plot(time_steps, x2_values, label='x2', color='r', linestyle='-')
plt.title('Dynamics of x1 and x2 over Time')
plt.xlabel('Time (t)')
plt.ylabel('Population Levels')
plt.legend()
plt.grid()
plt.show()
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
d = 0.1  # 交互作用强度
dt = 0.001  # 更小的时间步长
t_max = 50  # 最大时间
r_values = np.linspace(0, 2, 100)  # r的范围

# 存储结果
x1_results = []
x2_results = []

# 对每个r值进行计算
for r in r_values:
    # 重置状态，设定不同的初始条件
    y = np.array([1.0, 0.5])  # 设定初始条件
    t = 0
    # 使用RK4方法迭代
    while t < t_max:
        y = rk4_step(dynamic_system, y, dt, r, K, d)
        y = np.maximum(y, 0)  # 确保x1和x2都是非负的
        t += dt

    # 存储结果
    x1_results.append(y[0])
    x2_results.append(y[1])

# 将结果转换为NumPy数组（可选）
x1_results = np.array(x1_results)
x2_results = np.array(x2_results)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(r_values, x1_results, label='x1', color='b')
plt.plot(r_values, x2_results, label='x2', color='r')
plt.title('Dynamics of x1 and x2 as r varies')
plt.xlabel('r')
plt.ylabel('Population Levels')
plt.legend()
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 一些参数定义
r = 2 # 可以根据需要调整
alpha = 25  # 可以根据需要调整

# 定义不动点方程
def equilibrium(x, beta, r, alpha):
    return r * x * (1 - x/50) - (beta * x) / (alpha + x)

# 随着 beta 的变化，计算不动点
beta_values = np.linspace(40, 60, 1000)  # 在这里定义 beta 的范围
fixed_points = []

for beta in beta_values:
    # 使用 fsolve 计算不动点，给定初始猜测
    x_0 = 0.5  # 初始猜测
    x_fixed = fsolve(equilibrium, x_0, args=(beta, r, alpha))
    fixed_points.append(x_fixed[0])  # 提取不动点

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(beta_values, fixed_points)
plt.xlabel('β')
plt.grid()
plt.xlim(beta_values[0], beta_values[-1])
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

# 参数设置
r_B = 0.2
r_S = r_B/100.0
K_S = 4000
beta = 200
e_B = 0.002

# 函数定义
def f(S):
    return 4.0 * S  # 示例函数

def g(S):
    return S / 2.0  # 示例函数

# 模型方程
def model(y, t):
    B, S = y
    dBdt = r_B * B * (1 - B / f(S)) - (beta * B**2 / (g(S)**2 + B**2))
    dSdt = r_S * S * (1 - S / K_S) - e_B * B
    return [dBdt, dSdt]

# 不动点方程
def fixed_points(Y):
    B, S = Y
    dBdt = r_B * B * (1 - B / f(S)) - (beta * B**2 / (g(S)**2 + B**2))
    dSdt = r_S * S * (1 - S / K_S) - e_B * B
    return [dBdt, dSdt]

# 使用 fsolve 找到不动点
initial_guess = [1000, 1000]  # 初始猜测
fixed_point = fsolve(fixed_points, initial_guess)

# 创建网格
B = np.linspace(0, 10000, 20)
S = np.linspace(0, 5000, 20)
B_grid, S_grid = np.meshgrid(B, S)

# 计算向量场
DB, DS = np.zeros(B_grid.shape), np.zeros(S_grid.shape)
for i in range(B_grid.shape[0]):
    for j in range(B_grid.shape[1]):
        dBdt, dSdt = model((B_grid[i, j], S_grid[i, j]), 0)
        DB[i, j] = dBdt
        DS[i, j] = dSdt

# 绘制流线图
plt.figure(figsize=(10, 8))
plt.streamplot(B_grid, S_grid, DB, DS, color='r', linewidth=1, density=1)
plt.title('Streamline Plot of the Dynamic System')
plt.xlabel('B (Prey)')
plt.ylabel('S (Resource)')
plt.xlim(0, 10000)
plt.ylim(0, 5000)

# 标记不动点
plt.plot(fixed_point[0], fixed_point[1], 'bo')  # 使用蓝色圆点标记不动点
plt.annotate('Fixed Point ({:.2f}, {:.2f})'.format(fixed_point[0], fixed_point[1]),
             xy=(fixed_point[0], fixed_point[1]),
             xytext=(fixed_point[0] + 500, fixed_point[1] + 500),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.axhline(0, color='black', lw=0.5, ls='--')  # 添加 y=0 线
plt.axvline(0, color='black', lw=0.5, ls='--')  # 添加 x=0 线
plt.grid()
plt.show()
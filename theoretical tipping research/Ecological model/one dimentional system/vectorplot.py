import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 参数设置
r_x = 0.47  # x的生长率
r_y = 0.47  # y的生长率
K = 10.0   # 载荷

# 定义系统的方程
def system(X):
    x, y = X
    dxdt = r_x * x * (1 - (x - y) / K) - (x**2) / (1 + x**2)
    dydt = r_y * y * (1 - (y - x) / K) - (y**2) / (1 + y**2)
    return np.array([dxdt, dydt])

# 定义不动点方程
def fixed_points(X):
    return system(X)

# 找到不动点
initial_guess = [0.1, 5]  # 初始猜测
fixed_point = fsolve(fixed_points, initial_guess)

# 创建网格
x = np.linspace(-2, 12, 20)
y = np.linspace(-2, 12, 20)
X, Y = np.meshgrid(x, y)

# 计算向量场
DX, DY = np.zeros(X.shape), np.zeros(Y.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        dXdt, dYdt = system((X[i, j], Y[i, j]))
        DX[i, j] = dXdt
        DY[i, j] = dYdt

# 绘制流线图
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, DX, DY, color='r', linewidth=1, density=1)
plt.title('Streamline Plot of the Dynamic System')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-2, 12)
plt.ylim(-2, 12)

# 标记不动点
plt.plot(fixed_point[0], fixed_point[1], 'bo')  # 使用蓝色圆点标记不动点
plt.annotate('Fixed Point ({:.2f}, {:.2f})'.format(fixed_point[0], fixed_point[1]),
             xy=(fixed_point[0], fixed_point[1]),
             xytext=(fixed_point[0] + 1, fixed_point[1] + 1),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.axhline(0, color='black', lw=0.5, ls='--')  # 添加 y=0 线
plt.axvline(0, color='black', lw=0.5, ls='--')  # 添加 x=0 线
plt.grid()
plt.show()
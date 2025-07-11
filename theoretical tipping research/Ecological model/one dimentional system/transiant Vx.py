import numpy as np
import matplotlib.pyplot as plt

# 参数定义
r = 0.42
k = 10.0
beta = 1.0
alpha = 1.0

# 势能函数 V(x)
def V(x, r):
    return (r / 2) * x ** 2 - (r / 3) * (x ** 3 / k) + beta * x - beta * alpha * np.arctan(x / alpha)

# 计算势能函数的导数
def dV_dx(x, r):
    # 一阶导数
    return r * x - (r / k) * (x ** 2) + beta - beta * (1 / (1 + (x / alpha) ** 2)) * (1 / alpha)

def d2V_dx2(x, r):
    # 二阶导数
    return r - (2 * r / k) * x + beta * (2 * x / (alpha + x**2))

# 生成 x 轴数值
x_values = np.linspace(-2.5, 1, 4000)
V_values = V(x_values, r)

# 计算二阶导数并求出lambda
d2V_values = d2V_dx2(x_values, r)
lambda_values = -d2V_values**2  # lambda = -dV^2/dx^2

# 绘图
plt.figure(figsize=(10, 6))
sc = plt.scatter(x_values, V_values, c=lambda_values, cmap='viridis', marker='.')
plt.colorbar(sc, label=r'$\lambda = -\left(\frac{d^2V}{dx^2}\right)^2$')
plt.title("Potential Function")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.grid()
plt.show()
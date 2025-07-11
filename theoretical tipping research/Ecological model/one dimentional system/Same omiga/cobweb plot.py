import numpy as np
import matplotlib.pyplot as plt

# 定义参数
K = 10.0  # 环境承载能力
a = 1.0   # 参数a
b = 1.0   # 参数b
r = 0.50# 控制参数

# 定义动态方程
def f(x, r):
    return r * x * (1 - x / K) - (b ** 2 * x ** 2) / (a ** 2 + x ** 2)

# Cobweb图的初始值和迭代次数
x0 = 0.1  # 初始值
iterations = 5000  # 迭代次数

# 用于存储x_t的值
x_t = np.zeros(iterations)
x_t[0] = x0  # 设置初始值

# 迭代计算
for t in range(1, iterations):
    x_t[t] = f(x_t[t - 1], r)  # 更新x_{t+1}

# 生成x与f(x)值的范围
x = np.linspace(0, K, 400)
y = f(x, r)

# 绘制Cobweb图
plt.figure(figsize=(12, 8))
plt.plot(x, y, label='$x_{t+1} = f(x_t)$', color='blue')  # func curve
plt.plot(x, x, color='gray', linestyle='--', label='$y = x$ line')  # y=x line

# 绘制Cobweb线条
for i in range(iterations - 1):
    plt.plot([x_t[i], x_t[i]], [x_t[i], x_t[i + 1]], color='red')  # vertical line
    plt.plot([x_t[i], x_t[i + 1]], [x_t[i + 1], x_t[i + 1]], color='red')  # horizontal line

# 设置图形的细节
plt.title('Cobweb Diagram')
plt.xlabel('$x_t$')
plt.ylabel('$x_{t+1}$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.legend()
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 定义系统方程
def system(x, r_x, K):
    return r_x * x * (1 - x / K) - (x ** 2) / (1 + x ** 2)

# 设置参数
r_x = 0.47  # 可以根据要求修改
K = 10.0   # 可以根据要求修改

# 定义一个函数用于查找不动点
def find_fixed_points(r_x, K, initial_guess):
    fixed_point = fsolve(lambda x: system(x, r_x, K), initial_guess)
    return fixed_point

# 使用初始猜测找到不动点
initial_guesses = [0.1, 2.0, 5.0]  # 试多个初始猜测以找到不同的不动点
fixed_points = [find_fixed_points(r_x, K, guess)[0] for guess in initial_guesses]

# 创建x值的范围
x_values = np.linspace(0, K, 500)
dxdt_values = [system(x, r_x, K) for x in x_values]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_values, dxdt_values, label=r'$\frac{dx}{dt}$', color='blue')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

# 标示不动点
for point in fixed_points:
    plt.plot(point, 0, 'ro')  # 不动点
    plt.text(point, 0.1, f'({point:.2f}, 0)', fontsize=12, ha='center')

# 图形美化

plt.xlabel('x')
plt.ylabel(r'$\frac{dx}{dt}$')
plt.ylim(-1, 1.5)
plt.xlim(0, K)
plt.legend()
plt.grid()
plt.show()
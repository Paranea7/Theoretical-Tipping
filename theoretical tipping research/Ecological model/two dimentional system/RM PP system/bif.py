import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 系统参数
alpha = 0.19
beta = 0.03
gamma = 0.003
theta = 800
eta = 1.5
xi = 0.004
delta = 2.2

# 系统的微分方程
def system(t, vars, r0):
    p1, p2 = vars
    dp1dt = r0 * p1 * (1 - alpha * p1 / r0) * (p1 - beta) / (p1 + gamma) - theta * p1 * p2 / (eta + p1)
    dp2dt = xi * theta * p1 * p2 / (eta + p1) - delta * p2
    return [dp1dt, dp2dt]

# 定义时间范围
t_span = (0, 100)  # 0 到 10 时间单位
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 需要求解的时间点

# 绘制不动点随 r0 的变化
r0_values = np.linspace(0.1, 3, 600)  # r0的变化范围
p1_final_values = []
p2_final_values = []

initial_conditions = [10.81, 0.011]  # 初始条件

for r0 in r0_values:
    solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, args=(r0,))
    p1_final_values.append(solution.y[0][-1])  # 取最后的p1值
    p2_final_values.append(solution.y[1][-1])  # 取最后的p2值

# 画出不动点 p1
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)  # 创建两个子图的第一个
plt.plot(r0_values, p1_final_values, label='p1', color='blue')
plt.xlabel('r0')
plt.ylabel('p1')
plt.title('Fixed Point p1 as a Function of r0')
plt.grid()
plt.legend()

# 画出不动点 p2
plt.subplot(2, 1, 2)  # 创建两个子图的第二个
plt.plot(r0_values, p2_final_values, label='p2', color='orange')
plt.xlabel('r0')
plt.ylabel('p2')
plt.title('Fixed Point p2 as a Function of r0')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
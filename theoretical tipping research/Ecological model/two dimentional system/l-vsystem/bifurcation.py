import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 参数设置
K_S = 4000
beta = 200
e_B = 0.002


# 模型方程
def model(y, t, r_B, r_S):
    B, S = y
    dBdt = r_B * B * (1 - B / (4. * S)) - (beta * B ** 2 / ((S / 2.) ** 2 + B ** 2))
    dSdt = r_S * S * (1 - S / K_S) - e_B * B
    return [dBdt, dSdt]


# 计算稳态
def steady_state(r_B, r_S):
    # 初始条件
    B0 = 1.0
    S0 = 1.0
    y0 = [B0, S0]
    t = np.linspace(0, 100, 1000)

    # 解ODE
    sol = odeint(model, y0, t, args=(r_B, r_S))
    return sol[-1]  # 返回最后的稳态


# 设置r_B的范围
r_B_values = np.linspace(0.01, 20, 10000)  # 避免为0的情况
B_results = []
S_results = []

# 计算不同r_B下的稳态
for r_B in r_B_values:
    r_S = r_B / 100.  # 保持r_S与r_B的关系
    B, S = steady_state(r_B, r_S)
    B_results.append(B)
    S_results.append(S)

# 绘制分岔图
plt.figure(figsize=(12, 6))
plt.plot(r_B_values, B_results, label='B (Population B)', color='blue')
plt.plot(r_B_values, S_results, label='S (Resource S)', color='green')
plt.xlabel('r_B')
plt.ylabel('Steady State Values')
plt.title('B and S vs. r_B Bifurcation Diagram')
plt.legend()
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# 定义模型函数
def model(t, y, r_B):
    B, S = y
    r_S = r_B / 100.  # 资源 S 的增长率
    dBdt = r_B * B * (1 - B / (4 * S)) - (beta * B ** 2 / ((S / 2) ** 2 + B ** 2))
    dSdt = r_S * S * (1 - S / K_S) - e_B * B
    return [dBdt, dSdt]


# 参数设置
K_S = 4000  # 资源 S 的环境承载能力
beta = 200  # B 对 S 的竞争影响
e_B = 0.002  # B 对 S 的消耗率

# 设置初始条件
B0 = 10  # 初始物种 B 的数量
S0 = 3000  # 初始资源 S 的数量
y0 = [B0, S0]

# 设置时间范围
t_span = (0, 100)  # 从 t=0 到 t=100
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 在这个区间内采样500个点

# 存储最终 B 和 S 的值
B_final_values = []
S_final_values = []
r_B_range = np.linspace(0.01, 3, 300)  # r_B 的变化范围

# 遍历 r_B 的值
for r_B in r_B_range:
    # 求解ODE
    solution = solve_ivp(model, t_span, y0, t_eval=t_eval, args=(r_B,))

    # 提取最后的结果
    B_final_values.append(solution.y[0][-1])  # 最后时刻的 B 值
    S_final_values.append(solution.y[1][-1])  # 最后时刻的 S 值

# 在同一窗口中绘制 S-rS 图和 B-rB 图
plt.figure(figsize=(12, 10))

# 绘制 B-r_B 图
plt.subplot(2, 1, 1)  # 2行1列的第一个子图
plt.plot(r_B_range, B_final_values, marker='o', color='blue')
plt.xlabel('r_B (Growth Rate of Species B)')
plt.ylabel('B (Population of Species B)')
plt.title('Population of Species B vs r_B')
plt.grid()

# 绘制 S-r_S 图
plt.subplot(2, 1, 2)  # 2行1列的第二个子图
r_S_range = r_B_range / 100.  # 计算对应的 r_S
plt.plot(r_S_range, S_final_values, marker='o', color='green')
plt.xlabel('r_S (Growth Rate of Resource S)')
plt.ylabel('S (Resource S)')
plt.title('Resource S vs r_S')
plt.grid()

plt.tight_layout()
plt.show()
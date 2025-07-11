import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义周期性繁殖脉冲序列
period = 30  # 周期长度
pulse_1 = [0.1 if 4 <= i <= 27 else 1 for i in range(period)]  # 低电平（非繁殖期）: 4-27，高电平（繁殖期）: 1-3和28-30
pulse_2 = [0.1 if 4<= i <=17 else 1 for i in range(period)]

def k_strategy_with_pulse(y, t, s_base, K, alpha, beta, pulse):
    N, P = y
    current_time = int(t) % period  # 离散化时间以匹配脉冲周期
    s_1 = s_base * pulse_1[current_time]  # 动态调整繁殖率
    dNdt = s_1 * N * (1 - (N + alpha * P)/K)
    s_2 = s_base * pulse_2[current_time]
    dPdt = s_2 * P * (1 - (P + beta * N)/K)
    return [dNdt, dPdt]

# 参数设置
s_base, K, alpha, beta = 0.1, 5000, 0.6, 0.7
t = np.linspace(0, 2000, 20000)  # 延长模拟时间以观察周期
y0 = [100, 100]

# 求解方程
solution = odeint(k_strategy_with_pulse, y0, t, args=(s_base, K, alpha, beta, pulse_1))
N, P = solution.T

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(t, N, label="Species N")
plt.plot(t, P, label="Species P")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Manual")
plt.legend()
plt.grid(True)
plt.show()